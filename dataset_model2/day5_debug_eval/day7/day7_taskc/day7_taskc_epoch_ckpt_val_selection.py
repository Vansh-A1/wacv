#!/usr/bin/env python3
"""Day 7 Task C: validation-only checkpoint selection for saved Task B epoch ckpts."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PACKAGE_ROOT = Path("/home/projectwork/student_package")
DATASET_ROOT = PACKAGE_ROOT / "dataset_model2"
DAY5_DIR = DATASET_ROOT / "day5_debug_eval"
OUT_DIR = DAY5_DIR / "day7" / "day7_taskd"
PLOTS_DIR = OUT_DIR / "plots"

CKPT_DIR = DATASET_ROOT / "checkpoints" / "wacv_ea_day7_seed42"
SPLIT_CSV = DATASET_ROOT / "ref_splits_seed42" / "combined_kadid_tid_koniq_split_seed42.csv"

OUT_VAL_SCORES = OUT_DIR / "taskc_val_scores_all_epoch_ckpts.csv"
OUT_CORR_LONG = OUT_DIR / "taskc_val_correlations_long.csv"
OUT_SELECTION = OUT_DIR / "taskc_checkpoint_selection_table.csv"
OUT_SUMMARY = OUT_DIR / "taskc_summary.json"
OUT_REPORT = OUT_DIR / "taskc_report.txt"

PATH_REMAPS = {
    "/data/projectwork/swati_mam/kadid10k": "/data/projectwork/swati_mam/new model data/kadid10k",
}

sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "external"))
from external.dataloader import _resize_short_side, center_crop  # noqa: E402
from external.model import CVAEGenerator_v2  # noqa: E402
from score import sz_from_stats  # noqa: E402


def remap_path(path: str) -> str:
    for old, new in PATH_REMAPS.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


class ValImageDataset(Dataset):
    def __init__(self, paths: list[str], img_size: int = 256):
        self.paths = paths
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        pil = Image.open(self.paths[idx]).convert("RGB")
        pil = _resize_short_side(pil, self.img_size)
        pil = center_crop(pil, self.img_size)
        arr = (np.asarray(pil) / 255.0).astype("float32")
        return self.to_tensor(arr), idx


def epoch_from_path(path: Path) -> int:
    m = re.search(r"epoch_(\d+)\.pth$", path.name)
    if not m:
        raise ValueError(f"Not an epoch checkpoint path: {path}")
    return int(m.group(1))


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    if "mu_ref" not in ckpt or "Sigma_ref" not in ckpt:
        raise KeyError(f"{ckpt_path} has no embedded mu_ref/Sigma_ref; Task C requires the new ref saved in Task B epoch ckpts.")
    return model, ckpt["mu_ref"].to(device), ckpt["Sigma_ref"].to(device), img_size, ckpt


@torch.no_grad()
def score_paths(paths: list[str], ckpt_path: Path, device: torch.device, batch_size: int = 32) -> tuple[list[float], dict]:
    model, mu_ref, sigma_ref, img_size, ckpt = load_model(ckpt_path, device)
    loader = DataLoader(
        ValImageDataset(paths, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    scores: list[float] = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, mu, logvar = model(imgs)
        sz = sz_from_stats(mu, logvar, mu_ref, sigma_ref, sigma_t_max=1.0, mu_only=True)
        scores.extend(sz.detach().cpu().numpy().tolist())
    meta = {
        "checkpoint_path": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", epoch_from_path(ckpt_path))),
        "mu_ref_mean": float(ckpt["mu_ref"].float().mean().item()),
        "Sigma_ref_mean": float(ckpt["Sigma_ref"].float().mean().item()),
        "sz_mode": ckpt.get("sz_mode", "mu_only"),
        "sz_sigma_t_max": float(ckpt.get("sz_sigma_t_max", 1.0)),
    }
    return scores, meta


def spearman(x, y) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    sp = spearmanr(x[mask], y[mask])
    return float(sp.statistic), float(sp.pvalue), n


def load_validation_rows() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    df = df[(df["split"] == "val") & (df["dataset"].isin(["KADID-10k", "TID2013"]))].copy()
    df["distorted_path"] = df["distorted_path"].map(remap_path)
    df["severity_or_level"] = pd.to_numeric(df["severity_or_level"], errors="coerce")
    df["mos_or_dmos"] = pd.to_numeric(df["mos_or_dmos"], errors="coerce")
    return df.reset_index(drop=True)


def subset_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    dt = df["distortion_type"].astype(str).str.lower()
    if subset == "gaussian_blur":
        return (df["dataset"] == "KADID-10k") & (dt == "gaussian blur")
    if subset == "lens_blur":
        return (df["dataset"] == "KADID-10k") & (dt == "lens blur")
    if subset == "high_sharpen":
        return (df["dataset"] == "KADID-10k") & (dt == "high sharpen")
    if subset == "pixelate":
        return (df["dataset"] == "KADID-10k") & (dt == "pixelate")
    if subset == "pooled_mos":
        return df["dataset"].isin(["KADID-10k", "TID2013"]) & df["mos_or_dmos"].notna()
    raise ValueError(subset)


def compute_correlations(scored: pd.DataFrame, meta: dict) -> tuple[list[dict], dict]:
    corr_rows = []
    summary = {
        "Epoch": int(meta["epoch"]),
        "ckpt path": meta["checkpoint_path"],
    }
    severity_specs = [
        ("gaussian_blur", "blur SRCC"),
        ("lens_blur", "lens SRCC"),
        ("high_sharpen", "sharpen SRCC"),
        ("pixelate", "pixelation SRCC"),
    ]
    for subset, out_col in severity_specs:
        sub = scored[subset_mask(scored, subset)]
        rho, p, n = spearman(sub["E_A"], sub["severity_or_level"])
        summary[out_col] = rho
        corr_rows.append({
            "epoch": meta["epoch"],
            "ckpt_path": meta["checkpoint_path"],
            "metric": "Spearman(E_A, severity)",
            "subset": subset,
            "N": n,
            "spearman_rho": rho,
            "spearman_p": p,
            "desired_sign": "negative if higher severity lowers E_A",
        })

    pooled = scored[subset_mask(scored, "pooled_mos")]
    rho, p, n = spearman(pooled["E_A"], pooled["mos_or_dmos"])
    summary["pooled Spearman(E_A,MOS)"] = rho
    corr_rows.append({
        "epoch": meta["epoch"],
        "ckpt_path": meta["checkpoint_path"],
        "metric": "Spearman(E_A, MOS)",
        "subset": "pooled_KADID_TID_val",
        "N": n,
        "spearman_rho": rho,
        "spearman_p": p,
        "desired_sign": "negative if E_A is higher=worse and MOS is higher=better",
    })

    blur = summary["blur SRCC"]
    lens = summary["lens SRCC"]
    summary["selection_score"] = (-blur if np.isfinite(blur) else np.nan) + (-lens if np.isfinite(lens) else np.nan)
    return corr_rows, summary


def make_plots(selection: pd.DataFrame) -> list[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(selection["Epoch"], selection["selection_score"], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("selection_score = (-blur SRCC) + (-lens SRCC)")
    ax.set_title("Day 7 Task C validation checkpoint selection")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = PLOTS_DIR / "taskc_selection_score_vs_epoch.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in ["blur SRCC", "lens SRCC", "sharpen SRCC", "pixelation SRCC"]:
        ax.plot(selection["Epoch"], selection[col], marker="o", label=col)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("validation Spearman rho")
    ax.set_title("Day 7 Task C severity correlations")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = PLOTS_DIR / "taskc_severity_srcc_vs_epoch.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))
    return made


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpts = sorted(CKPT_DIR.glob("epoch_*.pth"), key=epoch_from_path)
    if not ckpts:
        raise FileNotFoundError(f"No epoch_*.pth checkpoints found in {CKPT_DIR}")

    val = load_validation_rows()
    val_paths = val["distorted_path"].astype(str).tolist()

    all_score_frames = []
    corr_rows = []
    selection_rows = []
    checkpoint_meta = []
    for ckpt_path in ckpts:
        scores, meta = score_paths(val_paths, ckpt_path, device)
        checkpoint_meta.append(meta)
        scored = val.copy()
        scored["epoch"] = meta["epoch"]
        scored["ckpt_path"] = meta["checkpoint_path"]
        scored["E_A"] = scores
        all_score_frames.append(scored)

        rows, summary = compute_correlations(scored, meta)
        corr_rows.extend(rows)
        selection_rows.append(summary)

    all_scores = pd.concat(all_score_frames, ignore_index=True)
    all_scores.to_csv(OUT_VAL_SCORES, index=False)
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_CORR_LONG, index=False)
    selection = pd.DataFrame(selection_rows).sort_values("Epoch").reset_index(drop=True)
    best_idx = selection["selection_score"].idxmax()
    selection["selected"] = "no"
    selection.loc[best_idx, "selected"] = "yes"
    selection.to_csv(OUT_SELECTION, index=False)
    plots = make_plots(selection)

    selected = selection.loc[best_idx].to_dict()
    summary = {
        "task": "Day 7 Task C: validation-only checkpoint selection",
        "output_dir": str(OUT_DIR),
        "checkpoint_dir": str(CKPT_DIR),
        "num_epoch_checkpoints": len(ckpts),
        "epochs_evaluated": [epoch_from_path(p) for p in ckpts],
        "device_used": str(device),
        "validation_rows": int(len(val)),
        "validation_by_dataset": val["dataset"].value_counts().to_dict(),
        "subsets": {
            "gaussian_blur": int(subset_mask(val, "gaussian_blur").sum()),
            "lens_blur": int(subset_mask(val, "lens_blur").sum()),
            "high_sharpen": int(subset_mask(val, "high_sharpen").sum()),
            "pixelate": int(subset_mask(val, "pixelate").sum()),
            "pooled_mos": int(subset_mask(val, "pooled_mos").sum()),
        },
        "selection_formula": "selection_score = (- SRCC_EA_sev_blur) + (- SRCC_EA_sev_lens)",
        "selected_checkpoint": selected,
        "checkpoint_meta": checkpoint_meta,
        "outputs": {
            "val_scores_csv": str(OUT_VAL_SCORES),
            "correlations_long_csv": str(OUT_CORR_LONG),
            "selection_table_csv": str(OUT_SELECTION),
            "summary_json": str(OUT_SUMMARY),
            "report_txt": str(OUT_REPORT),
            "plots": plots,
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "Day 7 Task C: validation-only epoch checkpoint selection",
        "",
        f"Checkpoint dir: {CKPT_DIR}",
        f"Epoch checkpoints evaluated: {summary['epochs_evaluated']}",
        f"Device used: {device}",
        "",
        "Validation data only:",
        f"- rows: {len(val)}",
        f"- by dataset: {summary['validation_by_dataset']}",
        f"- Gaussian blur rows: {summary['subsets']['gaussian_blur']}",
        f"- Lens blur rows: {summary['subsets']['lens_blur']}",
        f"- High sharpen rows: {summary['subsets']['high_sharpen']}",
        f"- Pixelate rows: {summary['subsets']['pixelate']}",
        f"- pooled MOS rows: {summary['subsets']['pooled_mos']}",
        "",
        "Selection formula:",
        "selection_score = (- SRCC_EA_sev_blur) + (- SRCC_EA_sev_lens)",
        "",
        "Selection table:",
        selection.to_string(index=False),
        "",
        "Selected checkpoint:",
        f"- epoch: {int(selected['Epoch'])}",
        f"- path: {selected['ckpt path']}",
        f"- selection_score: {float(selected['selection_score']):.6f}",
        "",
        "Outputs:",
        f"- {OUT_VAL_SCORES}",
        f"- {OUT_CORR_LONG}",
        f"- {OUT_SELECTION}",
        f"- {OUT_SUMMARY}",
        f"- {OUT_REPORT}",
        f"- {PLOTS_DIR}",
    ]
    OUT_REPORT.write_text("\n".join(report) + "\n")
    print(f"Saved Task C outputs under {OUT_DIR}")
    print(f"Selected epoch {int(selected['Epoch'])}: {selected['ckpt path']}")


if __name__ == "__main__":
    main()
