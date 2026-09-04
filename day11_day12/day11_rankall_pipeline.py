#!/usr/bin/env python3
"""Day 11 severity-pair rank-all pipeline.

Outputs live next to this script.
This script avoids KonIQ and pristine pairs for pair mining/training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg19


ROOT = Path("/home/projectwork/student_package")
OUT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "dataset_model2"
DEBUG_ROOT = DATASET_ROOT / "day5_debug_eval"
SPLIT_CSV = DATASET_ROOT / "ref_splits_seed42" / "combined_kadid_tid_koniq_split_seed42.csv"
INIT_CKPT = DATASET_ROOT / "checkpoints" / "wacv_ea_day7_seed42" / "epoch_0005.pth"
EH_CKPT = ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"
TASKD_HOLDOUT = DEBUG_ROOT / "day7" / "day7_taskD" / "taskD_holdout_scores.csv"
DAY8_COMPARE = DEBUG_ROOT / "day8" / "kadid_holdout_epoch5_v1_vs_v2_scores.csv"

PAIRS_CSV = OUT_ROOT / "pairs_severity_train.csv"
PAIR_STATS = OUT_ROOT / "pair_stats.txt"

LAMBDAS = [0.1, 1.0]
SAVE_EVERY = 5
DEFAULT_EPOCHS = 25
SEED = 42
MARGIN = 0.1
KADID = "KADID-10k"
TRAIN_DATASETS = ["KADID-10k", "TID2013"]
VAL_DATASETS = ["KADID-10k", "TID2013"]

PATH_REMAPS = {
    "/data/projectwork/swati_mam/kadid10k": "/data/projectwork/swati_mam/new model data/kadid10k",
}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))
from external.dataloader import _resize_short_side, center_crop  # noqa: E402
from external.model import CVAEGenerator_v2  # noqa: E402
from score import sz_from_stats  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def remap_path(path: str) -> str:
    for old, new in PATH_REMAPS.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    df["distorted_path_original"] = df["distorted_path"]
    df["distorted_path"] = df["distorted_path"].astype(str).map(remap_path)
    df["severity_or_level"] = pd.to_numeric(df["severity_or_level"], errors="coerce")
    df["mos_or_dmos"] = pd.to_numeric(df["mos_or_dmos"], errors="coerce")
    return df


class VGGPerceptualLoss(nn.Module):
    LAYER_IDS = (3, 8, 17, 26)

    def __init__(self, device: torch.device):
        super().__init__()
        try:
            vgg = vgg19(weights="DEFAULT").features.to(device).eval()
        except TypeError:
            vgg = vgg19(pretrained=True).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.slices = nn.ModuleList()
        prev = 0
        for idx in self.LAYER_IDS:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, idx + 1)]))
            prev = idx + 1
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = (recon - self.mean) / self.std
        y = (target - self.mean) / self.std
        loss = 0.0
        for sl in self.slices:
            x = sl(x)
            y = sl(y)
            loss = loss + F.mse_loss(x, y)
        return loss / float(len(self.slices))


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    if "mu_ref" not in ckpt or "Sigma_ref" not in ckpt:
        raise KeyError(f"{ckpt_path} missing mu_ref/Sigma_ref")
    return model, ckpt, ckpt["mu_ref"].to(device), ckpt["Sigma_ref"].to(device), img_size, latent_dim


def save_ckpt(path: Path, model, optimizer, epoch: int, config: dict, mu_ref, sigma_ref) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "seed": SEED,
            "mu_ref": mu_ref.detach().cpu(),
            "Sigma_ref": sigma_ref.detach().cpu(),
            "checkpoint_kind": "day11_rankall_epoch",
            "sz_mode": "mu_only",
            "sz_sigma_t_max": 1.0,
        },
        path,
    )


class PathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        path = str(self.df.loc[idx, "distorted_path"])
        pil = Image.open(path).convert("RGB")
        pil = _resize_short_side(pil, self.img_size)
        pil = center_crop(pil, self.img_size)
        arr = (np.asarray(pil) / 255.0).astype("float32")
        return self.to_tensor(arr), idx


class PairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, img_size: int):
        self.pairs = pairs.reset_index(drop=True)
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.pairs)

    def load_img(self, path: str):
        pil = Image.open(path).convert("RGB")
        pil = _resize_short_side(pil, self.img_size)
        pil = center_crop(pil, self.img_size)
        arr = (np.asarray(pil) / 255.0).astype("float32")
        return self.to_tensor(arr)

    def __getitem__(self, idx: int):
        row = self.pairs.loc[idx]
        return self.load_img(str(row["path_mild"])), self.load_img(str(row["path_severe"]))


@torch.no_grad()
def score_energy(df: pd.DataFrame, ckpt_path: Path, device: torch.device, batch_size: int = 64, workers: int = 4) -> list[float]:
    model, _ckpt, mu_ref, sigma_ref, img_size, _latent_dim = load_model(ckpt_path, device)
    loader = DataLoader(PathDataset(df, img_size), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    scores = []
    for imgs, _idx in loader:
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        _recon, mu, logvar = model(imgs)
        sz = sz_from_stats(mu, logvar, mu_ref, sigma_ref, sigma_t_max=1.0, mu_only=True)
        scores.extend(sz.detach().cpu().numpy().tolist())
    return scores


def generate_pairs(device: torch.device, ordered_rule: str = "all") -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_manifest()
    train = df[df["split"].eq("train") & df["dataset"].isin(TRAIN_DATASETS)].copy()
    train = train[train["ref_path"].ne("NA") & train["severity_or_level"].notna()].copy()
    if train["dataset"].eq("KONIQ-10k").any() or train["split"].ne("train").any():
        raise RuntimeError("Forbidden row leaked into pair generation")

    unique = train.drop_duplicates("distorted_path").copy().reset_index(drop=True)
    unique["E_H"] = score_energy(unique, EH_CKPT, device, batch_size=64 if device.type == "cuda" else 16, workers=4 if device.type == "cuda" else 0)
    eh_by_path = dict(zip(unique["distorted_path"], unique["E_H"]))

    rows = []
    for (dataset, ref_id, distortion_type), group in train.groupby(["dataset", "ref_id", "distortion_type"], sort=True):
        g = group.sort_values("severity_or_level")
        levels = sorted(g["severity_or_level"].dropna().unique().tolist())
        if len(levels) < 2:
            continue
        if ordered_rule == "adjacent":
            level_pairs = list(zip(levels[:-1], levels[1:]))
        else:
            level_pairs = [(a, b) for i, a in enumerate(levels) for b in levels[i + 1 :]]
        for mild, severe in level_pairs:
            mild_rows = g[g["severity_or_level"].eq(mild)]
            severe_rows = g[g["severity_or_level"].eq(severe)]
            if len(mild_rows) != 1 or len(severe_rows) != 1:
                continue
            m = mild_rows.iloc[0]
            s = severe_rows.iloc[0]
            eh_m = float(eh_by_path[m["distorted_path"]])
            eh_s = float(eh_by_path[s["distorted_path"]])
            rows.append(
                {
                    "ref_id": ref_id,
                    "distortion_type": distortion_type,
                    "path_mild": m["distorted_path"],
                    "path_severe": s["distorted_path"],
                    "sev_mild": float(mild),
                    "sev_severe": float(severe),
                    "abs_dEH": abs(eh_m - eh_s),
                    "dataset": dataset,
                    "image_id_mild": m["image_id"],
                    "image_id_severe": s["image_id"],
                }
            )
    pairs = pd.DataFrame(rows)
    pairs = pairs[["ref_id", "distortion_type", "path_mild", "path_severe", "sev_mild", "sev_severe", "abs_dEH", "dataset", "image_id_mild", "image_id_severe"]]
    pairs.to_csv(PAIRS_CSV, index=False)
    stats = [
        "Day 11 severity pair generation",
        f"rule: {ordered_rule}",
        f"pair_N: {len(pairs)}",
        f"datasets: {pairs['dataset'].value_counts().to_dict()}",
        f"distortion_type_count: {pairs['distortion_type'].nunique()}",
        "",
        "pairs by dataset/type:",
        pairs.groupby(["dataset", "distortion_type"]).size().to_string(),
        "",
        "Forbidden rows check:",
        f"KONIQ rows: {int(pairs['dataset'].eq('KONIQ-10k').sum())}",
        "split: train only by construction from split manifest",
        "MOS not used; MOS columns are not present in pairs CSV",
    ]
    PAIR_STATS.write_text("\n".join(stats) + "\n")


def rank_loss(e_mild: torch.Tensor, e_severe: torch.Tensor, margin: float) -> torch.Tensor:
    return F.softplus(margin - (e_mild - e_severe)).mean()


def train_lambda(lambda_rank: float, epochs: int, device: torch.device) -> None:
    run_dir = OUT_ROOT / "day11_rankall" / f"lambda_{lambda_rank}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not PAIRS_CSV.exists():
        raise FileNotFoundError(f"Run pairs first: {PAIRS_CSV}")
    pairs = pd.read_csv(PAIRS_CSV)
    model, ckpt, mu_ref, sigma_ref, img_size, latent_dim = load_model(INIT_CKPT, device)
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    start_epoch = 1
    last_path = ckpt_dir / "last.pth"
    if last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=False)
        if float((last.get("config") or {}).get("lambda_rank", lambda_rank)) == float(lambda_rank):
            model.load_state_dict(last["generator"])
            if "optimizer" in last:
                opt.load_state_dict(last["optimizer"])
            start_epoch = int(last.get("epoch", 0)) + 1
    pix_loss = nn.L1Loss()
    vgg_loss = VGGPerceptualLoss(device)
    loader = DataLoader(PairDataset(pairs, img_size), batch_size=8, shuffle=True, num_workers=4 if device.type == "cuda" else 0, pin_memory=device.type == "cuda", drop_last=False)
    metrics_path = run_dir / "train_log.csv"
    if not metrics_path.exists():
        with metrics_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lambda_rank", "margin", "L_total", "L_pix", "L_vgg", "L_KL", "L_rank", "time_sec"])
    config = dict(ckpt.get("config") or {})
    config.update({"day11": True, "lambda_rank": lambda_rank, "rank_margin": MARGIN, "pairs_csv": str(PAIRS_CSV), "init_from": str(INIT_CKPT), "epochs": epochs})

    if start_epoch > epochs:
        print(f"lambda={lambda_rank} already trained through epoch {start_epoch - 1}; target epochs={epochs}", flush=True)
        return

    for ep in range(start_epoch, epochs + 1):
        started = time.time()
        sums = {"tot": 0.0, "pix": 0.0, "vgg": 0.0, "kl": 0.0, "rank": 0.0}
        n = 0
        for mild, severe in loader:
            mild = mild.to(device, non_blocking=device.type == "cuda")
            severe = severe.to(device, non_blocking=device.type == "cuda")
            imgs = torch.cat([mild, severe], dim=0)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(imgs)
            lp = pix_loss(recon, imgs)
            lv = vgg_loss(recon, imgs)
            lk_raw = kl_loss(mu, logvar)
            lk = torch.clamp(lk_raw, min=1.0)
            b = mild.shape[0]
            mu_m, mu_s = mu[:b], mu[b:]
            lv_m, lv_s = logvar[:b], logvar[b:]
            e_m = sz_from_stats(mu_m, lv_m, mu_ref, sigma_ref, sigma_t_max=1.0, mu_only=True)
            e_s = sz_from_stats(mu_s, lv_s, mu_ref, sigma_ref, sigma_t_max=1.0, mu_only=True)
            lr = rank_loss(e_m, e_s, MARGIN)
            loss = lp + 0.1 * lv + lk + float(lambda_rank) * lr
            loss.backward()
            opt.step()
            sums["tot"] += float(loss.item())
            sums["pix"] += float(lp.item())
            sums["vgg"] += float(lv.item())
            sums["kl"] += float(lk_raw.item())
            sums["rank"] += float(lr.item())
            n += 1
        row = [ep, lambda_rank, MARGIN, sums["tot"] / n, sums["pix"] / n, sums["vgg"] / n, sums["kl"] / n, sums["rank"] / n, time.time() - started]
        with metrics_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        save_ckpt(ckpt_dir / "last.pth", model, opt, ep, config, mu_ref, sigma_ref)
        if ep % SAVE_EVERY == 0:
            save_ckpt(ckpt_dir / f"epoch_{ep:04d}.pth", model, opt, ep, config, mu_ref, sigma_ref)
        print(f"lambda={lambda_rank} epoch {ep}/{epochs} L_total={row[3]:.4f} L_rank={row[7]:.4f} time={row[8]:.1f}s", flush=True)


def finite_corr(x, y) -> tuple[float, float, int, float, float]:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return math.nan, math.nan, n, math.nan, math.nan
    sr = spearmanr(x, y)
    pr = pearsonr(x, y)
    return float(sr.statistic), float(sr.pvalue), n, float(pr.statistic), float(pr.pvalue)


def score_ckpt_frame(df: pd.DataFrame, ckpt_path: Path, device: torch.device) -> pd.DataFrame:
    out = df.copy()
    out["E_A"] = score_energy(out, ckpt_path, device, batch_size=64 if device.type == "cuda" else 16, workers=4 if device.type == "cuda" else 0)
    return out


def selection_score(row: dict) -> float:
    return -row["blur_SRCC"] - row["lens_SRCC"] - max(0.0, row["sharpen_SRCC"]) - max(0.0, row["pixelate_SRCC"])


def validate_select(device: torch.device) -> None:
    df = load_manifest()
    val = df[df["split"].eq("val") & df["dataset"].isin(VAL_DATASETS)].copy().reset_index(drop=True)
    rows = []
    for lambda_rank in LAMBDAS:
        ckpt_dir = OUT_ROOT / "day11_rankall" / f"lambda_{lambda_rank}" / "checkpoints"
        for ckpt_path in sorted(ckpt_dir.glob("epoch_*.pth")):
            m = re.search(r"epoch_(\d+).pth", ckpt_path.name)
            epoch = int(m.group(1)) if m else -1
            scored = score_ckpt_frame(val, ckpt_path, device)
            scored.to_csv(OUT_ROOT / "day11_rankall" / f"lambda_{lambda_rank}" / f"val_scores_epoch_{epoch:04d}.csv", index=False)
            row = {"lambda_rank": lambda_rank, "epoch": epoch, "ckpt_path": str(ckpt_path)}
            for dtype, key in [("Gaussian blur", "blur"), ("Lens blur", "lens"), ("High sharpen", "sharpen"), ("Pixelate", "pixelate")]:
                sub = scored[scored["dataset"].eq(KADID) & scored["distortion_type"].eq(dtype)]
                sr, p, n, _pr, _pp = finite_corr(sub["E_A"], sub["severity_or_level"])
                row[f"{key}_SRCC"] = sr
                row[f"{key}_N"] = n
            pooled = scored[scored["dataset"].isin(VAL_DATASETS)]
            sr, p, n, _pr, _pp = finite_corr(pooled["E_A"], pooled["mos_or_dmos"])
            row["pooled_EA_MOS_SRCC_report_only"] = sr
            row["pooled_N"] = n
            row["selection_score"] = selection_score(row)
            rows.append(row)
    sel = pd.DataFrame(rows)
    sel["selected"] = "no"
    for lambda_rank, group in sel.groupby("lambda_rank"):
        idx = group["selection_score"].idxmax()
        sel.loc[idx, "selected"] = "yes"
    sel.to_csv(OUT_ROOT / "val_selection_table.csv", index=False)


def evaluate_holdout(device: torch.device) -> None:
    sel = pd.read_csv(OUT_ROOT / "val_selection_table.csv")
    df = load_manifest()
    hold = df[df["split"].eq("holdout") & df["dataset"].eq(KADID)].copy().reset_index(drop=True)
    taskd = pd.read_csv(TASKD_HOLDOUT)
    base = taskd[taskd["dataset"].eq(KADID)][["image_id", "E_H", "E_A", "z_H", "z_A", "Qz_AmH", "Qz_HmA", "mos_or_dmos", "distortion_type", "severity_or_level"]].rename(
        columns={"E_A": "S0_E_A_ep05", "z_A": "S0_z_A_ep05", "Qz_AmH": "S0_Qz_AmH_ep05", "Qz_HmA": "S0_Qz_HmA_ep05"}
    )
    score_frames = []
    for _, selected in sel[sel["selected"].eq("yes")].iterrows():
        lambda_rank = float(selected["lambda_rank"])
        ckpt_path = Path(str(selected["ckpt_path"]))
        scored = score_ckpt_frame(hold, ckpt_path, device)
        val_scored = pd.read_csv(OUT_ROOT / "day11_rankall" / f"lambda_{lambda_rank}" / f"val_scores_epoch_{int(selected['epoch']):04d}.csv")
        mean_a = float(val_scored[val_scored["dataset"].eq(KADID)]["E_A"].mean())
        std_a = float(val_scored[val_scored["dataset"].eq(KADID)]["E_A"].std(ddof=0))
        tmp = scored[["image_id", "E_A"]].rename(columns={"E_A": f"rankall_E_A_lambda_{lambda_rank}"})
        tmp[f"rankall_z_A_lambda_{lambda_rank}"] = (tmp[f"rankall_E_A_lambda_{lambda_rank}"] - mean_a) / std_a
        score_frames.append((lambda_rank, tmp))
    out = base.copy()
    for lambda_rank, tmp in score_frames:
        out = out.merge(tmp, on="image_id", how="left", validate="one_to_one")
        zcol = f"rankall_z_A_lambda_{lambda_rank}"
        out[f"rankall_Qz_AmH_lambda_{lambda_rank}"] = out[zcol] - out["z_H"]
        out[f"rankall_Qz_HmA_lambda_{lambda_rank}"] = out["z_H"] - out[zcol]
    out.to_csv(OUT_ROOT / "holdout_rankall_scores.csv", index=False)

    metric_rows = []
    for arm, col in [("S0 ep05 (Day-8)", "S0_Qz_AmH_ep05"), ("E_H vs MOS", "E_H")]:
        sr, p, n, pr, pp = finite_corr(out[col], out["mos_or_dmos"])
        metric_rows.append({"arm": arm, "metric": "Q_z^(A-H) vs MOS" if arm.startswith("S0") else "E_H vs MOS", "N": n, "SRCC": sr, "Pearson": pr})
    for lambda_rank, _tmp in score_frames:
        qcol = f"rankall_Qz_AmH_lambda_{lambda_rank}"
        ecol = f"rankall_E_A_lambda_{lambda_rank}"
        for metric, col, target, dtype in [
            ("Q_z^(A-H) vs MOS", qcol, "mos_or_dmos", None),
            ("Blur E_A--sev", ecol, "severity_or_level", "Gaussian blur"),
            ("Sharpen E_A--sev", ecol, "severity_or_level", "High sharpen"),
            ("Pixelate E_A--sev", ecol, "severity_or_level", "Pixelate"),
        ]:
            sub = out if dtype is None else out[out["distortion_type"].eq(dtype)]
            sr, p, n, pr, pp = finite_corr(sub[col], sub[target])
            metric_rows.append({"arm": f"Rank-all lambda_{lambda_rank}", "metric": metric, "N": n, "SRCC": sr, "Pearson": pr})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT_ROOT / "holdout_comparison_table.csv", index=False)
    report = [
        "Day 11 report",
        "",
        "Goal: severity pairwise ranking, no MOS in training, no E_H gate, no pristine pairs.",
        f"Pairs CSV: {PAIRS_CSV}",
        f"Pair stats: {PAIR_STATS}",
        "",
        "Validation selection:",
        sel.to_string(index=False),
        "",
        "Holdout comparison:",
        metrics.to_string(index=False),
    ]
    (OUT_ROOT / "day11_report.txt").write_text("\n".join(report) + "\n")
    (OUT_ROOT / "day11_summary.json").write_text(json.dumps({"selected": sel[sel["selected"].eq("yes")].to_dict("records"), "outputs": ["pairs_severity_train.csv", "val_selection_table.csv", "holdout_rankall_scores.csv", "holdout_comparison_table.csv", "day11_report.txt"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["pairs", "train", "select", "holdout", "all"])
    parser.add_argument("--lambda_rank", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--pair_rule", choices=["all", "adjacent"], default="all")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    set_seed(SEED)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.cmd in ["pairs", "all"]:
        generate_pairs(device, ordered_rule=args.pair_rule)
    if args.cmd in ["train", "all"]:
        lambdas = [args.lambda_rank] if args.lambda_rank is not None else LAMBDAS
        for lam in lambdas:
            train_lambda(float(lam), args.epochs, device)
    if args.cmd in ["select", "all"]:
        validate_select(device)
    if args.cmd in ["holdout", "all"]:
        evaluate_holdout(device)


if __name__ == "__main__":
    main()
