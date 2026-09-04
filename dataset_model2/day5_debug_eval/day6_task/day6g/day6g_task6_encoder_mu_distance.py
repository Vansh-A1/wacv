#!/usr/bin/env python3
"""Day 6 Task 6: save encoder mu vectors and distances to mu_ref for 200 images."""
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
from scipy.stats import kendalltau, pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PACKAGE_ROOT = Path("/home/projectwork/student_package")
DAY5_DIR = PACKAGE_ROOT / "dataset_model2" / "day5_debug_eval"
OUT_DIR = DAY5_DIR / "day6g"
PLOTS_DIR = OUT_DIR / "plots"

TASK4_200 = DAY5_DIR / "day6e" / "kadid_200_old_vs_day5_scores.csv"
EH_CKPT = PACKAGE_ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"
REF_STATS = PACKAGE_ROOT / "runs" / "hr_combined_ft1" / "reference_stats.pt"

OUT_MU = OUT_DIR / "task6_200_encoder_mu_vectors.csv"
OUT_CORR = OUT_DIR / "task6_distance_correlations.csv"
OUT_JSON = OUT_DIR / "task6_encoder_mu_distance_summary.json"
OUT_TXT = OUT_DIR / "task6_encoder_mu_distance_report.txt"

sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "external"))
from external.dataloader import _resize_short_side, center_crop  # noqa: E402
from external.model import CVAEGenerator_v2  # noqa: E402


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


def load_eh_model(device: torch.device):
    ckpt = torch.load(EH_CKPT, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    return model, img_size


@torch.no_grad()
def encode_mu(paths: list[str], device: torch.device, batch_size: int = 64) -> np.ndarray:
    model, img_size = load_eh_model(device)
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


def corr_row(x_name: str, y_name: str, x, y) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return {
            "x": x_name, "y": y_name, "N": int(mask.sum()),
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "pearson_r": np.nan, "pearson_p": np.nan,
            "kendall_tau": np.nan, "kendall_p": np.nan,
        }
    sp = spearmanr(x[mask], y[mask])
    pr = pearsonr(x[mask], y[mask])
    kt = kendalltau(x[mask], y[mask])
    return {
        "x": x_name,
        "y": y_name,
        "N": int(mask.sum()),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "kendall_tau": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
    }


def make_plot(df: pd.DataFrame, x_col: str, y_col: str, fname: str, title: str) -> str:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df[x_col], df[y_col], s=18, alpha=0.65)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = PLOTS_DIR / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(TASK4_200)
    if len(base) != 200:
        raise ValueError(f"Expected 200 fixed rows from Task 4, found {len(base)} in {TASK4_200}")

    ref = torch.load(REF_STATS, map_location="cpu", weights_only=False)
    mu_ref = ref["mu_ref"].float().view(-1).numpy()
    if mu_ref.shape[0] != 100:
        raise ValueError(f"Expected 100-D mu_ref, found shape {mu_ref.shape}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mus = encode_mu(base["distorted_path"].astype(str).tolist(), device)
    if mus.shape != (200, 100):
        raise ValueError(f"Expected encoded mu shape (200, 100), got {mus.shape}")

    d = np.linalg.norm(mus - mu_ref.reshape(1, -1), axis=1)

    out = base.copy()
    out["d_euclidean_to_mu_ref"] = d
    for i in range(mus.shape[1]):
        out[f"mu_{i:03d}"] = mus[:, i]
    out.to_csv(OUT_MU, index=False)

    corr_rows = [
        corr_row("d_euclidean_to_mu_ref", "E_H", out["d_euclidean_to_mu_ref"], out["E_H"]),
        corr_row("d_euclidean_to_mu_ref", "MOS", out["d_euclidean_to_mu_ref"], out["mos_or_dmos"]),
    ]
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_CORR, index=False)

    plots = [
        make_plot(out, "d_euclidean_to_mu_ref", "E_H", "task6_d_vs_E_H.png", "Euclidean d to mu_ref vs E_H"),
        make_plot(out, "d_euclidean_to_mu_ref", "mos_or_dmos", "task6_d_vs_MOS.png", "Euclidean d to mu_ref vs MOS"),
    ]

    d_stats = {
        "N": int(len(d)),
        "mean": float(np.mean(d)),
        "std": float(np.std(d, ddof=1)),
        "min": float(np.min(d)),
        "median": float(np.median(d)),
        "max": float(np.max(d)),
    }
    summary = {
        "task": "Task 6: encoder mean vectors and Euclidean distance to pristine mu_ref",
        "inputs": {
            "fixed_200_task4_csv": str(TASK4_200),
            "eh_checkpoint": str(EH_CKPT),
            "reference_stats_pt": str(REF_STATS),
        },
        "preprocess": "PIL RGB -> resize short side to >=256 if needed -> center 256x256 crop -> float RGB [0,1] -> ToTensor; no mean/std normalization.",
        "mu_shape": list(mus.shape),
        "mu_ref_shape": list(mu_ref.shape),
        "distance_definition": "d = Euclidean norm ||mu_image - mu_ref||_2 in the 100-D latent mean space.",
        "d_stats": d_stats,
        "correlations": corr_rows,
        "outputs": {
            "mu_vectors_csv": str(OUT_MU),
            "correlations_csv": str(OUT_CORR),
            "summary_json": str(OUT_JSON),
            "report_txt": str(OUT_TXT),
            "plots": plots,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    c_eh = corr_rows[0]
    c_mos = corr_rows[1]
    report = f"""Day 6 Task 6: encoder mu distance to pristine mu_ref

Input 200-image list:
{TASK4_200}

Model/checkpoint:
{EH_CKPT}

Reference stats:
{REF_STATS}

Preprocess:
PIL RGB -> resize short side to >=256 if needed -> center 256x256 crop -> float RGB [0,1] -> ToTensor; no mean/std normalization.

Saved encoder mean:
- one 100-float mu vector per image
- encoded mu array shape: {list(mus.shape)}
- mu_ref shape: {list(mu_ref.shape)}

Distance:
d = ||mu_image - mu_ref||_2

d statistics across 200 images:
- mean: {d_stats['mean']:.8f}
- std: {d_stats['std']:.8f}
- min: {d_stats['min']:.8f}
- median: {d_stats['median']:.8f}
- max: {d_stats['max']:.8f}

Correlations:
- d vs E_H: Spearman rho = {c_eh['spearman_rho']:+.6f}, Pearson r = {c_eh['pearson_r']:+.6f}, N = {c_eh['N']}
- d vs MOS: Spearman rho = {c_mos['spearman_rho']:+.6f}, Pearson r = {c_mos['pearson_r']:+.6f}, N = {c_mos['N']}

Interpretation note:
E_H is a Mahalanobis-style distance using mu_ref and Sigma_ref. This task uses plain Euclidean distance to mu_ref, so it should be highly related but not numerically identical to E_H.

Outputs:
- {OUT_MU}
- {OUT_CORR}
- {OUT_JSON}
- {OUT_TXT}
- {PLOTS_DIR}
"""
    OUT_TXT.write_text(report)
    print(f"Saved Task 6 outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
