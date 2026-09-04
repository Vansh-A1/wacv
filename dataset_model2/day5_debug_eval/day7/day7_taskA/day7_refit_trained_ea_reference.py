#!/usr/bin/env python3
"""Day 7: refit E_A reference stats using the trained lambda0 encoder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PACKAGE_ROOT = Path("/home/projectwork/student_package")
DATASET_ROOT = PACKAGE_ROOT / "dataset_model2"
DAY5_DIR = DATASET_ROOT / "day5_debug_eval"
OUT_DIR = DAY5_DIR / "day7"
PLOTS_DIR = OUT_DIR / "plots"

EA_CKPT = DATASET_ROOT / "checkpoints" / "wacv_degradation_from_scratch_lambda0_seed42" / "best.pth"
OLD_EA_REF = DATASET_ROOT / "runs" / "wacv_degradation_from_scratch_lambda0_seed42" / "reference_stats.pt"
PRISTINE_REF = PACKAGE_ROOT / "runs" / "hr_combined_ft1" / "reference_stats.pt"
REF_PATHS_CSV = DAY5_DIR / "ea_reference_set_diagnostics.csv"
TASK4_200_CANDIDATES = [
    DAY5_DIR / "day6e" / "kadid_200_old_vs_day5_scores.csv",
    DAY5_DIR / "day6_task" / "day6e" / "kadid_200_old_vs_day5_scores.csv",
]
DAY5_HOLDOUT = DAY5_DIR / "day5_holdout_scores.csv"

OUT_REF_PT = OUT_DIR / "ea_trained_encoder_refit_reference_stats.pt"
OUT_REF_PATHS = OUT_DIR / "day7_reference_paths_used.csv"
OUT_MU_CSV = OUT_DIR / "day7_reference_mu_vectors.csv"
OUT_COMPARE = OUT_DIR / "day7_reference_stats_comparison.csv"
OUT_200 = OUT_DIR / "day7_200_diagnostics_with_new_ref.csv"
OUT_200_CORR = OUT_DIR / "day7_200_new_ref_correlations.csv"
OUT_JSON = OUT_DIR / "day7_summary.json"
OUT_TXT = OUT_DIR / "day7_report.txt"

sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "external"))
from external.dataloader import _resize_short_side, center_crop  # noqa: E402
from external.model import CVAEGenerator_v2  # noqa: E402
from score import sz_from_stats  # noqa: E402


class ImagePathDataset(Dataset):
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


def load_ea_model(device: torch.device):
    ckpt = torch.load(EA_CKPT, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    return model, img_size, ckpt


@torch.no_grad()
def encode_paths(paths: list[str], model, img_size: int, device: torch.device, batch_size: int = 64) -> np.ndarray:
    loader = DataLoader(
        ImagePathDataset(paths, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    mus = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, mu, _ = model(imgs)
        mus.append(mu.detach().cpu().numpy())
    return np.concatenate(mus, axis=0)


def tensor_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {
        "shape": list(x.shape),
        "finite": bool(np.isfinite(x).all()),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def compare_vectors(name: str, a_name: str, a: np.ndarray, b_name: str, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return {
        "vector": name,
        "left": a_name,
        "right": b_name,
        "dim": int(a.size),
        "cosine_similarity": cosine(a, b),
        "l2_distance": float(np.linalg.norm(a - b)),
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
        "max_abs_diff": float(np.max(np.abs(a - b))),
        "left_mean": float(np.mean(a)),
        "right_mean": float(np.mean(b)),
    }


def corr_row(x_name: str, y_name: str, x, y) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return {"x": x_name, "y": y_name, "N": int(mask.sum()), "spearman_rho": np.nan, "pearson_r": np.nan}
    sp = spearmanr(x[mask], y[mask])
    pr = pearsonr(x[mask], y[mask])
    return {
        "x": x_name,
        "y": y_name,
        "N": int(mask.sum()),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
    }


def load_fixed_200_diagnostic_rows() -> tuple[pd.DataFrame, str]:
    for candidate in TASK4_200_CANDIDATES:
        if candidate.is_file():
            return pd.read_csv(candidate), str(candidate)

    holdout = pd.read_csv(DAY5_HOLDOUT)
    fixed = holdout[(holdout["dataset"] == "KADID-10k") & (holdout["has_mos"] == True)].copy()
    fixed = fixed.sort_values("image_id").head(200).reset_index(drop=True)
    fixed = fixed.rename(columns={"mos_or_dmos": "mos_or_dmos"})
    fixed["D_old"] = np.nan
    keep = [
        "image_id", "ref_id", "distorted_path", "ref_path", "distortion_type",
        "severity_or_level", "mos_or_dmos", "D_old", "E_H",
    ]
    return fixed[keep], f"reconstructed first 200 KADID holdout rows from {DAY5_HOLDOUT}"


def make_plots(compare_df: pd.DataFrame, diag_200: pd.DataFrame) -> list[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    made = []

    fig, ax = plt.subplots(figsize=(8, 4))
    mu_rows = compare_df[compare_df["vector"] == "mu_ref"]
    ax.bar(mu_rows["right"], mu_rows["l2_distance"])
    ax.set_ylabel("L2 distance from new trained E_A mu_ref")
    ax.set_title("Day 7 reference mean-vector separation")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p = PLOTS_DIR / "day7_mu_ref_l2_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(diag_200["E_A_new_ref"], diag_200["E_H"], s=18, alpha=0.65)
    ax.set_xlabel("E_A_new_ref")
    ax.set_ylabel("E_H")
    ax.set_title("200 KADID images: new E_A ref score vs E_H")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = PLOTS_DIR / "day7_200_EA_new_ref_vs_EH.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    return made


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, img_size, ckpt = load_ea_model(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    paths_df = pd.read_csv(REF_PATHS_CSV)
    paths_df = paths_df[(paths_df["loaded"] == True) & (paths_df["pool"].isin(["KADID-10k", "KONIQ-10k", "TID2013"]))].copy()
    paths_df = paths_df.head(2000).reset_index(drop=True)
    paths_df.to_csv(OUT_REF_PATHS, index=False)
    paths = paths_df["path"].astype(str).tolist()

    mus = encode_paths(paths, model, img_size, device)
    mu_ref_new = mus.mean(axis=0).astype("float32")
    sigma_ref_new = np.var(mus, axis=0).clip(min=1e-8).astype("float32")

    torch.save(
        {
            "mu_ref": torch.from_numpy(mu_ref_new),
            "Sigma_ref": torch.from_numpy(sigma_ref_new),
            "source_checkpoint": str(EA_CKPT),
            "source_checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "source_paths_csv": str(OUT_REF_PATHS),
            "N": int(len(paths)),
            "preprocess": "RGB -> resize short side >=256 if needed -> center 256 crop -> float [0,1] -> ToTensor; no flip.",
            "note": "Reference stats refit after training using the trained E_A encoder; original checkpoint was not modified.",
        },
        OUT_REF_PT,
    )

    mu_cols = pd.DataFrame(mus, columns=[f"mu_{i:03d}" for i in range(mus.shape[1])])
    pd.concat([paths_df[["index", "path", "pool"]].reset_index(drop=True), mu_cols], axis=1).to_csv(OUT_MU_CSV, index=False)

    old_ea = torch.load(OLD_EA_REF, map_location="cpu", weights_only=False)
    h_ref = torch.load(PRISTINE_REF, map_location="cpu", weights_only=False)
    mu_old_ea = old_ea["mu_ref"].float().numpy()
    sig_old_ea = old_ea["Sigma_ref"].float().numpy()
    mu_h = h_ref["mu_ref"].float().numpy()
    sig_h = h_ref["Sigma_ref"].float().numpy()

    compare_rows = [
        compare_vectors("mu_ref", "new_trained_E_A_ref", mu_ref_new, "old_epoch0_E_A_ref", mu_old_ea),
        compare_vectors("mu_ref", "new_trained_E_A_ref", mu_ref_new, "pristine_E_H_ref", mu_h),
        compare_vectors("Sigma_ref", "new_trained_E_A_ref", sigma_ref_new, "old_epoch0_E_A_ref", sig_old_ea),
        compare_vectors("Sigma_ref", "new_trained_E_A_ref", sigma_ref_new, "pristine_E_H_ref", sig_h),
    ]
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(OUT_COMPARE, index=False)

    task4, task4_source = load_fixed_200_diagnostic_rows()
    task4_paths = task4["distorted_path"].astype(str).tolist()
    task4_mus = encode_paths(task4_paths, model, img_size, device)
    with torch.no_grad():
        mu_t = torch.from_numpy(task4_mus).to(device)
        dummy_logvar = torch.zeros_like(mu_t)
        ea_new = sz_from_stats(
            mu_t,
            dummy_logvar,
            torch.from_numpy(mu_ref_new).to(device),
            torch.from_numpy(sigma_ref_new).to(device),
            sigma_t_max=1.0,
            mu_only=True,
        ).cpu().numpy()
    diag_200 = task4.copy()
    diag_200["E_A_new_ref"] = ea_new
    diag_200.to_csv(OUT_200, index=False)

    corr_rows = [
        corr_row("E_A_new_ref", "E_H", diag_200["E_A_new_ref"], diag_200["E_H"]),
        corr_row("E_A_new_ref", "MOS", diag_200["E_A_new_ref"], diag_200["mos_or_dmos"]),
        corr_row("D_old", "E_A_new_ref", diag_200["D_old"], diag_200["E_A_new_ref"]),
    ]
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_200_CORR, index=False)

    plots = make_plots(compare_df, diag_200)

    summary = {
        "task": "Day 7: refit E_A reference stats after training",
        "used_checkpoint": str(EA_CKPT),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "weights_reset": False,
        "encoder_mode": "eval + no_grad; parameters require_grad=False during this script",
        "reference_source": {
            "input_csv": str(REF_PATHS_CSV),
            "output_paths_csv": str(OUT_REF_PATHS),
            "N": int(len(paths)),
            "pool_counts": paths_df["pool"].value_counts().to_dict(),
            "not_pristine_df2k": True,
        },
        "diagnostic_200_source": task4_source,
        "preprocess": "RGB -> resize short side >=256 if needed -> center 256 crop -> float [0,1] -> ToTensor; no flip.",
        "new_reference_stats": {
            "mu_ref_A_new": tensor_summary(mu_ref_new),
            "Sigma_ref_A_new": tensor_summary(sigma_ref_new),
        },
        "comparisons": compare_rows,
        "diagnostic_200_correlations": corr_rows,
        "outputs": {
            "new_reference_stats_pt": str(OUT_REF_PT),
            "reference_paths_used": str(OUT_REF_PATHS),
            "reference_mu_vectors": str(OUT_MU_CSV),
            "reference_comparison_csv": str(OUT_COMPARE),
            "diagnostic_200_csv": str(OUT_200),
            "diagnostic_200_correlations_csv": str(OUT_200_CORR),
            "summary_json": str(OUT_JSON),
            "report_txt": str(OUT_TXT),
            "plots": plots,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "Day 7: trained E_A reference-stat refit",
        "",
        f"Checkpoint loaded: {EA_CKPT}",
        f"Checkpoint epoch: {int(ckpt.get('epoch', -1))}",
        "Weights reset: no",
        "Mode: model.eval(), torch.no_grad(), parameters frozen for this script",
        "",
        "Reference images used:",
        f"- Source CSV: {REF_PATHS_CSV}",
        f"- N: {len(paths)}",
        f"- Pool counts: {paths_df['pool'].value_counts().to_dict()}",
        "- Safety: distorted/authentic train only; no holdout, no pristine DF2K.",
        "",
        "Preprocess:",
        "- RGB, resize short side to >=256 if needed, center 256 crop, float [0,1], no flip.",
        "",
        "New reference stats:",
        f"- mu_ref_A_new shape/min/max/mean: {list(mu_ref_new.shape)}, {mu_ref_new.min():.8f}, {mu_ref_new.max():.8f}, {mu_ref_new.mean():.8f}",
        f"- Sigma_ref_A_new shape/min/max/mean: {list(sigma_ref_new.shape)}, {sigma_ref_new.min():.12g}, {sigma_ref_new.max():.12g}, {sigma_ref_new.mean():.12g}",
        "",
        "Reference comparisons:",
        compare_df.to_string(index=False),
        "",
        "200-image diagnostic with new ref:",
        f"Diagnostic source: {task4_source}",
        corr_df.to_string(index=False),
        "",
        "Important note:",
        "- The original checkpoint and original reference_stats.pt were not modified.",
        "- Use the new ref for later tasks from this file:",
        f"  {OUT_REF_PT}",
        "",
        "Outputs:",
    ]
    report.extend(f"- {v}" for v in summary["outputs"].values() if not isinstance(v, list))
    report.extend(f"- {p}" for p in plots)
    OUT_TXT.write_text("\n".join(report) + "\n")
    print(f"Saved Day 7 outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
