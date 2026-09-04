#!/usr/bin/env python3
"""Task Day 12 / day11b pristine-rank pipeline.

This trains the pristine WACV encoder with severity ranking:
for mild x and severe y, enforce E_H(y) >= E_H(x) + margin.

Outputs stay under:
/home/projectwork/student_package/day11_day12/task12/day11b_pristine_rank
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

ROOT = Path("/home/projectwork/student_package")
DAY11_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DAY11_ROOT))
import day11_rankall_pipeline as base  # noqa: E402

OUT_ROOT = DAY11_ROOT / "task12" / "day11b_pristine_rank"
PAIRS_CSV = DAY11_ROOT / "pairs_severity_train.csv"
INIT_CKPT = ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"
TASKD_HOLDOUT = ROOT / "dataset_model2" / "day5_debug_eval" / "day7" / "day7_taskD" / "taskD_holdout_scores.csv"

LAMBDAS = [0.1, 1.0]
SAVE_EVERY = 5
DEFAULT_EPOCHS = 25
SEED = 42
MARGIN = 0.1
KADID = "KADID-10k"
VAL_DATASETS = ["KADID-10k", "TID2013"]


def finite_corr(x, y):
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


def pristine_rank_loss(e_mild: torch.Tensor, e_severe: torch.Tensor, margin: float) -> torch.Tensor:
    return F.softplus(margin - (e_severe - e_mild)).mean()


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
            "checkpoint_kind": "day12_pristine_rank_epoch",
            "sz_mode": "mu_only",
            "sz_sigma_t_max": 1.0,
        },
        path,
    )


def train_lambda(lambda_rank: float, epochs: int, device: torch.device) -> None:
    run_dir = OUT_ROOT / f"lambda_{lambda_rank}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not PAIRS_CSV.exists():
        raise FileNotFoundError(f"Missing Day11 pair CSV: {PAIRS_CSV}")

    pairs = pd.read_csv(PAIRS_CSV)
    forbidden = []
    if any("mos" in c.lower() for c in pairs.columns):
        forbidden.append("MOS column present")
    if int(pairs["dataset"].eq("KONIQ-10k").sum()) != 0:
        forbidden.append("KonIQ rows present")
    if int((pairs["sev_mild"] >= pairs["sev_severe"]).sum()) != 0:
        forbidden.append("severity order errors")
    if forbidden:
        raise RuntimeError("; ".join(forbidden))

    model, ckpt, mu_ref, sigma_ref, img_size, _latent_dim = base.load_model(INIT_CKPT, device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    start_epoch = 1
    last_path = ckpt_dir / "last.pth"
    if last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=False)
        if float((last.get("config") or {}).get("lambda_rank", lambda_rank)) == float(lambda_rank):
            model.load_state_dict(last["generator"])
            if "optimizer" in last:
                optimizer.load_state_dict(last["optimizer"])
            start_epoch = int(last.get("epoch", 0)) + 1
    pix_loss = nn.L1Loss()
    vgg_loss = base.VGGPerceptualLoss(device)
    loader = DataLoader(
        base.PairDataset(pairs, img_size),
        batch_size=8,
        shuffle=True,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    config = dict(ckpt.get("config") or {})
    config.update(
        {
            "day12_pristine_rank": True,
            "lambda_rank": lambda_rank,
            "rank_margin": MARGIN,
            "rank_direction": "severe_E_H_higher_than_mild_E_H",
            "pairs_csv": str(PAIRS_CSV),
            "init_from": str(INIT_CKPT),
            "epochs": epochs,
            "mos_used_in_training": False,
            "koniq_used_in_training": False,
        }
    )

    log_path = run_dir / "train_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lambda_rank", "margin", "L_total", "L_pix", "L_vgg", "L_KL_raw", "L_rank", "time_sec", "ckpt_path"])

    if start_epoch > epochs:
        print(f"lambda={lambda_rank} already trained through epoch {start_epoch - 1}; target epochs={epochs}", flush=True)
        return

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        sums = {"total": 0.0, "pix": 0.0, "vgg": 0.0, "kl": 0.0, "rank": 0.0}
        n = 0
        for mild, severe in loader:
            mild = mild.to(device, non_blocking=device.type == "cuda")
            severe = severe.to(device, non_blocking=device.type == "cuda")
            imgs = torch.cat([mild, severe], dim=0)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(imgs)
            lp = pix_loss(recon, imgs)
            lv = vgg_loss(recon, imgs)
            lk_raw = base.kl_loss(mu, logvar)
            lk = torch.clamp(lk_raw, min=1.0)
            bsz = mild.shape[0]
            e_mild = base.sz_from_stats(mu[:bsz], logvar[:bsz], mu_ref, sigma_ref, sigma_t_max=1.0, mu_only=True)
            e_severe = base.sz_from_stats(mu[bsz:], logvar[bsz:], mu_ref, sigma_ref, sigma_t_max=1.0, mu_only=True)
            lr = pristine_rank_loss(e_mild, e_severe, MARGIN)
            loss = lp + 0.1 * lv + lk + float(lambda_rank) * lr
            loss.backward()
            optimizer.step()

            sums["total"] += float(loss.item())
            sums["pix"] += float(lp.item())
            sums["vgg"] += float(lv.item())
            sums["kl"] += float(lk_raw.item())
            sums["rank"] += float(lr.item())
            n += 1

        ckpt_path = ckpt_dir / "last.pth"
        save_ckpt(ckpt_path, model, optimizer, epoch, config, mu_ref, sigma_ref)
        saved_path = ""
        if epoch % SAVE_EVERY == 0:
            saved_path = str(ckpt_dir / f"epoch_{epoch:04d}.pth")
            save_ckpt(Path(saved_path), model, optimizer, epoch, config, mu_ref, sigma_ref)
        row = [
            epoch,
            lambda_rank,
            MARGIN,
            sums["total"] / n,
            sums["pix"] / n,
            sums["vgg"] / n,
            sums["kl"] / n,
            sums["rank"] / n,
            time.time() - started,
            saved_path,
        ]
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"lambda={lambda_rank} epoch {epoch}/{epochs} L_total={row[3]:.4f} L_rank={row[7]:.4f} ckpt={saved_path}", flush=True)


def score_frame(df: pd.DataFrame, ckpt_path: Path, device: torch.device) -> pd.DataFrame:
    out = df.copy()
    out["E_H_ranked"] = base.score_energy(out, ckpt_path, device, batch_size=64 if device.type == "cuda" else 16, workers=4 if device.type == "cuda" else 0)
    return out


def selection_score(row: dict) -> float:
    return -row["blur_SRCC"] - row["lens_SRCC"] - max(0.0, row["sharpen_SRCC"]) - max(0.0, row["pixelate_SRCC"])


def validate_select(device: torch.device) -> None:
    df = base.load_manifest()
    val = df[df["split"].eq("val") & df["dataset"].isin(VAL_DATASETS)].copy().reset_index(drop=True)
    rows = []
    for lambda_rank in LAMBDAS:
        ckpt_dir = OUT_ROOT / f"lambda_{lambda_rank}" / "checkpoints"
        for ckpt_path in sorted(ckpt_dir.glob("epoch_*.pth")):
            epoch = int(re.search(r"epoch_(\d+).pth", ckpt_path.name).group(1))
            scored = score_frame(val, ckpt_path, device)
            scored.to_csv(OUT_ROOT / f"lambda_{lambda_rank}" / f"val_scores_epoch_{epoch:04d}.csv", index=False)
            row = {"lambda_rank": lambda_rank, "epoch": epoch, "ckpt_path": str(ckpt_path)}
            for dtype, key in [("Gaussian blur", "blur"), ("Lens blur", "lens"), ("High sharpen", "sharpen"), ("Pixelate", "pixelate")]:
                sub = scored[scored["dataset"].eq(KADID) & scored["distortion_type"].eq(dtype)]
                sr, _p, n, _pr, _pp = finite_corr(sub["E_H_ranked"], sub["severity_or_level"])
                row[f"{key}_SRCC"] = sr
                row[f"{key}_N"] = n
            pooled = scored[scored["dataset"].isin(VAL_DATASETS)]
            sr, _p, n, pr, _pp = finite_corr(pooled["E_H_ranked"], pooled["mos_or_dmos"])
            row["pooled_EH_MOS_SRCC_report_only"] = sr
            row["pooled_EH_MOS_Pearson_report_only"] = pr
            row["pooled_N"] = n
            row["selection_score"] = selection_score(row)
            rows.append(row)

    table = pd.DataFrame(rows)
    table["selected"] = "no"
    for lambda_rank, group in table.groupby("lambda_rank"):
        table.loc[group["selection_score"].idxmax(), "selected"] = "yes"
    table.to_csv(OUT_ROOT / "val_selection_table.csv", index=False)


def evaluate_holdout(device: torch.device) -> None:
    selection = pd.read_csv(OUT_ROOT / "val_selection_table.csv")
    df = base.load_manifest()
    hold = df[df["split"].eq("holdout") & df["dataset"].eq(KADID)].copy().reset_index(drop=True)
    baseline = pd.read_csv(TASKD_HOLDOUT)
    baseline = baseline[baseline["dataset"].eq(KADID)][["image_id", "dataset", "ref_id", "distortion_type", "severity_or_level", "mos_or_dmos", "E_H"]].copy()
    out = baseline.copy()

    scored_cols = []
    for _, row in selection[selection["selected"].eq("yes")].iterrows():
        lambda_rank = float(row["lambda_rank"])
        ckpt_path = Path(str(row["ckpt_path"]))
        scored = score_frame(hold, ckpt_path, device)[["image_id", "E_H_ranked"]].rename(columns={"E_H_ranked": f"ranked_E_H_lambda_{lambda_rank}"})
        out = out.merge(scored, on="image_id", how="left", validate="one_to_one")
        scored_cols.append((f"Pristine+rank lambda_{lambda_rank}", f"ranked_E_H_lambda_{lambda_rank}"))

    out.to_csv(OUT_ROOT / "holdout_metrics.csv", index=False)

    rows = []
    arms = [("Unranked E_H (Day-8)", "E_H")] + scored_cols
    for arm, col in arms:
        for metric, target, dtype in [
            ("E_H vs MOS", "mos_or_dmos", None),
            ("Blur E_H--sev", "severity_or_level", "Gaussian blur"),
            ("Sharpen E_H--sev", "severity_or_level", "High sharpen"),
            ("Pixelate E_H--sev", "severity_or_level", "Pixelate"),
        ]:
            sub = out if dtype is None else out[out["distortion_type"].eq(dtype)]
            sr, p, n, pr, pp = finite_corr(sub[col], sub[target])
            rows.append({"arm": arm, "metric": metric, "N": n, "SRCC": sr, "SRCC_p": p, "Pearson": pr, "Pearson_p": pp})
    compare = pd.DataFrame(rows)
    compare.to_csv(OUT_ROOT / "compare_vs_unranked_EH.csv", index=False)

    report = [
        "Task Day 12 / day11b pristine-rank report",
        "",
        "Rank direction: for mild x and severe y, enforce E_H(y) >= E_H(x) + margin.",
        f"Init pristine checkpoint: {INIT_CKPT}",
        f"Pair CSV reused from Day11: {PAIRS_CSV}",
        "Pair CSV contains no MOS columns and no KonIQ rows by validation before training.",
        "",
        "Validation selection table:",
        selection.to_string(index=False),
        "",
        "KADID holdout comparison:",
        compare.to_string(index=False),
    ]
    (OUT_ROOT / "day11b_report.txt").write_text("\n".join(report) + "\n")
    (OUT_ROOT / "day11b_summary.json").write_text(
        json.dumps(
            {
                "init_checkpoint": str(INIT_CKPT),
                "pairs_csv": str(PAIRS_CSV),
                "rank_direction": "severe_E_H_higher_than_mild_E_H",
                "outputs": ["val_selection_table.csv", "holdout_metrics.csv", "compare_vs_unranked_EH.csv", "day11b_report.txt"],
                "selected": selection[selection["selected"].eq("yes")].to_dict("records"),
            },
            indent=2,
        )
        + "\n"
    )


def write_preflight() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIRS_CSV) if PAIRS_CSV.exists() else pd.DataFrame()
    preflight = {
        "output_root": str(OUT_ROOT),
        "init_checkpoint": str(INIT_CKPT),
        "pair_csv": str(PAIRS_CSV),
        "pair_rows": int(len(pairs)),
        "pair_columns": list(pairs.columns),
        "mos_column_in_pairs": bool(any("mos" in c.lower() for c in pairs.columns)),
        "koniq_rows_in_pairs": int(pairs["dataset"].eq("KONIQ-10k").sum()) if "dataset" in pairs else None,
        "bad_severity_order_rows": int((pairs["sev_mild"] >= pairs["sev_severe"]).sum()) if {"sev_mild", "sev_severe"}.issubset(pairs.columns) else None,
        "lambda_values": LAMBDAS,
        "margin": MARGIN,
        "default_epochs": DEFAULT_EPOCHS,
    }
    (OUT_ROOT / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["preflight", "train", "select", "holdout", "all"])
    parser.add_argument("--lambda_rank", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    base.set_seed(SEED)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.cmd in ["preflight", "all"]:
        write_preflight()
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
