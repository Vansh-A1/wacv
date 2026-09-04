#!/usr/bin/env python3
"""Day 9 Task B: evaluate ep05/ep10 GMM artifact-energy models."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PACKAGE_ROOT = Path("/home/projectwork/student_package")
DATASET_ROOT = PACKAGE_ROOT / "dataset_model2"
DAY5_ROOT = DATASET_ROOT / "day5_debug_eval"
DAY7_ROOT = DAY5_ROOT / "day7"
DAY8_ROOT = DAY5_ROOT / "day8"
DAY9A_ROOT = DAY5_ROOT / "day9" / "day9_taskA"
OUT_DIR = DAY5_ROOT / "day9" / "day9_taskb"
CACHE_DIR = OUT_DIR / "encoded_mu_cache"

SPLIT_CSV = DATASET_ROOT / "ref_splits_seed42" / "combined_kadid_tid_koniq_split_seed42.csv"
TASKD_HOLDOUT = DAY7_ROOT / "day7_taskD" / "taskD_holdout_scores.csv"
TASKD_VAL_STATS = DAY7_ROOT / "day7_taskD" / "taskD_val_normalization_stats.csv"
DAY8_COMPARE = DAY8_ROOT / "kadid_holdout_epoch5_v1_vs_v2_scores.csv"

CKPTS = {
    "ep05": DATASET_ROOT / "checkpoints" / "wacv_ea_day7_seed42" / "epoch_0005.pth",
    "ep10": DATASET_ROOT / "checkpoints" / "wacv_ea_day7_seed42" / "epoch_0010.pth",
}
GMM_DIRS = {
    "ep05": DAY9A_ROOT / "day9_gmm" / "ep05",
    "ep10": DAY9A_ROOT / "day9_gmm" / "ep10",
}
K_VALUES = [2, 4, 8, 16]
KADID = "KADID-10k"
VAL_DATASETS = ["KADID-10k", "TID2013"]
WITHIN_TYPES = ["Gaussian blur", "Lens blur", "High sharpen", "Pixelate"]

PATH_REMAPS = {
    "/data/projectwork/swati_mam/kadid10k": "/data/projectwork/swati_mam/new model data/kadid10k",
}

sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "external"))
from external.dataloader import _resize_short_side, center_crop  # noqa: E402
from external.model import CVAEGenerator_v2  # noqa: E402


def remap_path(path: str) -> str:
    for old, new in PATH_REMAPS.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


class ImageDataset(Dataset):
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


def require_inputs() -> None:
    paths = [SPLIT_CSV, TASKD_HOLDOUT, TASKD_VAL_STATS, DAY8_COMPARE]
    paths.extend(CKPTS.values())
    for tag, gmm_dir in GMM_DIRS.items():
        paths.append(gmm_dir / "mu_train.npy")
        paths.append(gmm_dir / "mu_train_ids.csv")
        for k in K_VALUES:
            paths.append(gmm_dir / f"gmm_K{k:02d}.pt")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    df["distorted_path_original"] = df["distorted_path"]
    df["distorted_path"] = df["distorted_path"].astype(str).map(remap_path)
    df["severity_or_level"] = pd.to_numeric(df["severity_or_level"], errors="coerce")
    df["mos_or_dmos"] = pd.to_numeric(df["mos_or_dmos"], errors="coerce")
    return df


def load_encoder(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    meta = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "img_size": img_size,
        "latent_dim": latent_dim,
    }
    return model, img_size, latent_dim, meta


@torch.no_grad()
def encode_mu(df: pd.DataFrame, tag: str, split_name: str, ckpt_path: Path, device: torch.device) -> tuple[np.ndarray, dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mu_path = CACHE_DIR / f"{tag}_{split_name}_mu.npy"
    ids_path = CACHE_DIR / f"{tag}_{split_name}_ids.csv"
    if mu_path.exists() and ids_path.exists():
        cached_ids = pd.read_csv(ids_path)
        if len(cached_ids) == len(df) and cached_ids["image_id"].astype(str).tolist() == df["image_id"].astype(str).tolist():
            mu = np.load(mu_path)
            return mu, {"cache": "hit", "mu_path": str(mu_path), "ids_path": str(ids_path), "N": int(len(df))}

    model, img_size, latent_dim, meta = load_encoder(ckpt_path, device)
    batch_size = 64 if device.type == "cuda" else 16
    workers = 4 if device.type == "cuda" else 0
    loader = DataLoader(
        ImageDataset(df, img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda" and workers > 0),
    )
    mu = np.zeros((len(df), latent_dim), dtype=np.float32)
    started = time.time()
    for imgs, idxs in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        _recon, mu_batch, _logvar = model(imgs)
        mu[idxs.numpy()] = mu_batch.detach().cpu().numpy().astype(np.float32)
    np.save(mu_path, mu)
    df.to_csv(ids_path, index=False)
    meta.update(
        {
            "cache": "miss",
            "mu_path": str(mu_path),
            "ids_path": str(ids_path),
            "N": int(len(df)),
            "mu_shape": list(mu.shape),
            "encode_seconds": float(time.time() - started),
            "batch_size": batch_size,
            "workers": workers,
            "device": str(device),
        }
    )
    return mu, meta


def gmm_energies(mu: np.ndarray, gmm_path: Path) -> tuple[np.ndarray, np.ndarray]:
    gmm = torch.load(gmm_path, map_location="cpu", weights_only=False)
    weights = np.asarray(gmm["weights"], dtype=np.float64)
    means = np.asarray(gmm["means"], dtype=np.float64)
    cov = np.asarray(gmm["covariances"], dtype=np.float64)
    x = np.asarray(mu, dtype=np.float64)

    diff = x[:, None, :] - means[None, :, :]
    log_comp = (
        np.log(weights[None, :])
        - 0.5 * np.sum(np.log(2.0 * np.pi * cov[None, :, :]) + (diff * diff) / cov[None, :, :], axis=2)
    )
    soft_nll = -logsumexp(log_comp, axis=1)
    hard_min_nll = np.min(-log_comp, axis=1)
    return soft_nll.astype(np.float32), hard_min_nll.astype(np.float32)


def corr(x, y) -> tuple[float, float, int]:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return math.nan, math.nan, n
    sp = spearmanr(x, y)
    return float(sp.statistic), float(sp.pvalue), n


def pearson(x, y) -> tuple[float, float, int]:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return math.nan, math.nan, n
    pr = pearsonr(x, y)
    return float(pr.statistic), float(pr.pvalue), n


def add_gmm_columns(base: pd.DataFrame, mu: np.ndarray, tag: str) -> pd.DataFrame:
    out = base.copy()
    for k in K_VALUES:
        soft, hard = gmm_energies(mu, GMM_DIRS[tag] / f"gmm_K{k:02d}.pt")
        out[f"E_A_GMM_soft_{tag}_K{k:02d}"] = soft
        out[f"E_A_GMM_min_{tag}_K{k:02d}"] = hard
    return out


def val_metrics(val_scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    selected: dict[str, int] = {}
    for tag in CKPTS:
        best_k = None
        best_score = -np.inf
        for k in K_VALUES:
            for variant in ["soft", "min"]:
                col = f"E_A_GMM_{variant}_{tag}_K{k:02d}"
                row = {
                    "epoch_tag": tag,
                    "K": k,
                    "energy_variant": variant,
                    "energy_column": col,
                }
                for distortion_type, short in [
                    ("Gaussian blur", "blur"),
                    ("Lens blur", "lens"),
                    ("High sharpen", "sharpen"),
                    ("Pixelate", "pixelate"),
                ]:
                    sub = val_scores[
                        val_scores["dataset"].eq(KADID)
                        & val_scores["distortion_type"].eq(distortion_type)
                    ]
                    rho, p, n = corr(sub[col], sub["severity_or_level"])
                    row[f"{short}_SRCC"] = rho
                    row[f"{short}_p"] = p
                    row[f"{short}_N"] = n
                pooled = val_scores[val_scores["dataset"].isin(VAL_DATASETS)]
                rho, p, n = corr(pooled[col], pooled["mos_or_dmos"])
                row["pooled_EA_MOS_SRCC_report_only"] = rho
                row["pooled_EA_MOS_p"] = p
                row["pooled_EA_MOS_N"] = n
                if variant == "soft":
                    row["K_score"] = (
                        -row["blur_SRCC"]
                        - row["lens_SRCC"]
                        - max(0.0, row["sharpen_SRCC"])
                        - max(0.0, row["pixelate_SRCC"])
                    )
                    if row["K_score"] > best_score:
                        best_score = float(row["K_score"])
                        best_k = k
                else:
                    row["K_score"] = math.nan
                rows.append(row)
        if best_k is None:
            raise RuntimeError(f"No K selected for {tag}")
        selected[tag] = int(best_k)
    metrics = pd.DataFrame(rows)
    return metrics, selected


def write_k_selection(metrics: pd.DataFrame, selected: dict[str, int]) -> None:
    for tag, best_k in selected.items():
        rows = metrics[(metrics["epoch_tag"].eq(tag)) & (metrics["energy_variant"].eq("soft"))].copy()
        rows["selected_K"] = rows["K"].eq(best_k).map({True: "yes", False: "no"})
        rows.to_csv(OUT_DIR / f"k_selection_{tag}.csv", index=False)


def gmm_val_stats(val_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag in CKPTS:
        for k in K_VALUES:
            for variant in ["soft", "min"]:
                col = f"E_A_GMM_{variant}_{tag}_K{k:02d}"
                for dataset, group in val_scores.groupby("dataset", sort=True):
                    rows.append(
                        {
                            "epoch_tag": tag,
                            "K": k,
                            "energy_variant": variant,
                            "dataset": dataset,
                            "N_val": int(len(group)),
                            "mean_E_A_GMM": float(group[col].mean()),
                            "std_E_A_GMM": float(group[col].std(ddof=0)),
                        }
                    )
    return pd.DataFrame(rows)


def build_holdout_scores(holdout_base: pd.DataFrame, val_stats: pd.DataFrame) -> pd.DataFrame:
    taskd = pd.read_csv(TASKD_HOLDOUT)
    kadid_taskd = taskd[taskd["dataset"].eq(KADID)].copy()
    day8 = pd.read_csv(DAY8_COMPARE)
    ep10_cols = day8[["image_id", "E_A_v2", "z_A_v2", "Qz_AmH_v2", "Qz_HmA_v2"]].copy()

    out = holdout_base.merge(
        kadid_taskd[
            [
                "image_id",
                "E_H",
                "E_A",
                "Qz_AmH",
                "Qz_HmA",
                "z_H",
                "z_A",
            ]
        ].rename(
            columns={
                "E_A": "S0_E_A_ep05",
                "z_A": "S0_z_A_ep05",
                "Qz_AmH": "S0_Qz_AmH_ep05",
                "Qz_HmA": "S0_Qz_HmA_ep05",
            }
        ),
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        ep10_cols.rename(
            columns={
                "E_A_v2": "S0_E_A_ep10",
                "z_A_v2": "S0_z_A_ep10",
                "Qz_AmH_v2": "S0_Qz_AmH_ep10",
                "Qz_HmA_v2": "S0_Qz_HmA_ep10",
            }
        ),
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    stats_lookup = val_stats.set_index(["epoch_tag", "K", "energy_variant", "dataset"])
    for tag in CKPTS:
        for k in K_VALUES:
            for variant in ["soft", "min"]:
                ea_col = f"E_A_GMM_{variant}_{tag}_K{k:02d}"
                z_col = f"z_A_GMM_{variant}_{tag}_K{k:02d}"
                q_amh = f"Qz_AmH_GMM_{variant}_{tag}_K{k:02d}"
                q_hma = f"Qz_HmA_GMM_{variant}_{tag}_K{k:02d}"
                st = stats_lookup.loc[(tag, k, variant, KADID)]
                std = float(st["std_E_A_GMM"])
                mean = float(st["mean_E_A_GMM"])
                out[z_col] = (out[ea_col] - mean) / std if std > 0 else np.nan
                out[q_amh] = out[z_col] - out["z_H"]
                out[q_hma] = out["z_H"] - out[z_col]
    return out


def holdout_metrics(holdout: pd.DataFrame, selected: dict[str, int]) -> pd.DataFrame:
    rows = []
    for tag in CKPTS:
        for k in K_VALUES:
            for variant in ["soft", "min"]:
                ea_col = f"E_A_GMM_{variant}_{tag}_K{k:02d}"
                q_amh = f"Qz_AmH_GMM_{variant}_{tag}_K{k:02d}"
                q_hma = f"Qz_HmA_GMM_{variant}_{tag}_K{k:02d}"
                for score_name, col, target, distortion_type in [
                    ("E_A_GMM vs MOS", ea_col, "mos_or_dmos", None),
                    ("Qz_AmH_GMM vs MOS", q_amh, "mos_or_dmos", None),
                    ("Qz_HmA_GMM vs MOS", q_hma, "mos_or_dmos", None),
                    ("Blur E_A_GMM vs severity", ea_col, "severity_or_level", "Gaussian blur"),
                    ("Sharpen E_A_GMM vs severity", ea_col, "severity_or_level", "High sharpen"),
                    ("Pixelate E_A_GMM vs severity", ea_col, "severity_or_level", "Pixelate"),
                ]:
                    sub = holdout if distortion_type is None else holdout[holdout["distortion_type"].eq(distortion_type)]
                    sr, sr_p, n = corr(sub[col], sub[target])
                    pr, pr_p, _ = pearson(sub[col], sub[target])
                    rows.append(
                        {
                            "epoch_tag": tag,
                            "K": k,
                            "selected_K_for_epoch": selected[tag],
                            "is_selected_K": k == selected[tag],
                            "energy_variant": variant,
                            "metric": score_name,
                            "subset": "KADID holdout" if distortion_type is None else f"KADID holdout {distortion_type}",
                            "N": n,
                            "SRCC": sr,
                            "SRCC_p": sr_p,
                            "Pearson": pr,
                            "Pearson_p": pr_p,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    require_inputs()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    manifest = load_manifest()
    val = manifest[manifest["split"].eq("val") & manifest["dataset"].isin(VAL_DATASETS)].copy().reset_index(drop=True)
    holdout = manifest[manifest["split"].eq("holdout") & manifest["dataset"].eq(KADID)].copy().reset_index(drop=True)

    missing = [p for p in pd.concat([val["distorted_path"], holdout["distorted_path"]]).astype(str).tolist() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image files: {len(missing)}. First: {missing[0]}")

    run_meta = {
        "task": "Day 9 Task B - GMM energy evaluation",
        "device": str(device),
        "val_N": int(len(val)),
        "holdout_N": int(len(holdout)),
        "inputs": {
            "split_csv": str(SPLIT_CSV),
            "taskD_holdout": str(TASKD_HOLDOUT),
            "day8_epoch5_vs_v2": str(DAY8_COMPARE),
            "day9_taskA_gmm_root": str(DAY9A_ROOT / "day9_gmm"),
        },
        "energy_definition": {
            "soft": "negative log likelihood under diagonal GMM: -logsumexp(log pi_k + log N(mu | mean_k, diag cov_k))",
            "hard_min": "minimum per-component negative weighted log likelihood: min_k[-(log pi_k + log N(mu | mean_k, diag cov_k))]",
        },
        "encodings": {},
    }

    val_scores = val.copy()
    holdout_scores = holdout.copy()
    for tag, ckpt_path in CKPTS.items():
        val_mu, val_meta = encode_mu(val, tag, "val_kadid_tid", ckpt_path, device)
        holdout_mu, holdout_meta = encode_mu(holdout, tag, "kadid_holdout", ckpt_path, device)
        run_meta["encodings"][tag] = {"val": val_meta, "holdout": holdout_meta}
        val_scores = add_gmm_columns(val_scores, val_mu, tag)
        holdout_scores = add_gmm_columns(holdout_scores, holdout_mu, tag)

    val_scores.to_csv(OUT_DIR / "val_all8_scores.csv", index=False)
    metrics, selected = val_metrics(val_scores)
    metrics.to_csv(OUT_DIR / "val_all8_metrics.csv", index=False)
    write_k_selection(metrics, selected)

    stats = gmm_val_stats(val_scores)
    stats.to_csv(OUT_DIR / "gmm_val_normalization_stats.csv", index=False)

    holdout_joined = build_holdout_scores(holdout_scores, stats)
    holdout_joined.to_csv(OUT_DIR / "holdout_all8_scores.csv", index=False)
    h_metrics = holdout_metrics(holdout_joined, selected)
    h_metrics.to_csv(OUT_DIR / "holdout_metrics.csv", index=False)

    selected_rows = h_metrics[h_metrics["is_selected_K"] & h_metrics["energy_variant"].eq("soft")].copy()
    selected_rows.to_csv(OUT_DIR / "holdout_selected_soft_metrics.csv", index=False)

    run_meta["selected_K"] = selected
    run_meta["outputs"] = {
        "val_all8_scores": str(OUT_DIR / "val_all8_scores.csv"),
        "val_all8_metrics": str(OUT_DIR / "val_all8_metrics.csv"),
        "k_selection_ep05": str(OUT_DIR / "k_selection_ep05.csv"),
        "k_selection_ep10": str(OUT_DIR / "k_selection_ep10.csv"),
        "gmm_val_normalization_stats": str(OUT_DIR / "gmm_val_normalization_stats.csv"),
        "holdout_all8_scores": str(OUT_DIR / "holdout_all8_scores.csv"),
        "holdout_metrics": str(OUT_DIR / "holdout_metrics.csv"),
        "holdout_selected_soft_metrics": str(OUT_DIR / "holdout_selected_soft_metrics.csv"),
        "eval_report": str(OUT_DIR / "eval_report.txt"),
        "eval_summary": str(OUT_DIR / "eval_summary.json"),
    }
    (OUT_DIR / "eval_summary.json").write_text(json.dumps(run_meta, indent=2))

    report = [
        "Day 9 Task B completed",
        "",
        "Baseline/ref-only instruction followed: frozen E_H and S0 single-Gaussian baseline scores were joined from existing Day 7/Day 8 CSVs, not recomputed.",
        f"Device used for new GMM encoding: {device}",
        f"Val rows: {len(val)} (KADID+TID val)",
        f"KADID holdout rows: {len(holdout)}",
        "",
        "K selection on val, soft GMM energy only:",
    ]
    for tag in CKPTS:
        report.append("")
        report.append(f"{tag}:")
        report.append(pd.read_csv(OUT_DIR / f"k_selection_{tag}.csv").to_string(index=False))
    report.extend(
        [
            "",
            "Selected soft-GMM holdout metrics:",
            selected_rows.to_string(index=False),
            "",
            "Energy definitions:",
            run_meta["energy_definition"]["soft"],
            run_meta["energy_definition"]["hard_min"],
            "",
        ]
    )
    (OUT_DIR / "eval_report.txt").write_text("\n".join(report))
    (OUT_DIR / "DONE.txt").write_text(
        "Day 9 Task B DONE\n"
        f"Output dir: {OUT_DIR}\n"
        f"Selected K ep05: {selected['ep05']}\n"
        f"Selected K ep10: {selected['ep10']}\n"
    )

    print(f"Day 9 Task B complete: {OUT_DIR}")
    print(f"Selected K: {selected}")
    print(selected_rows.to_string(index=False))


if __name__ == "__main__":
    main()
