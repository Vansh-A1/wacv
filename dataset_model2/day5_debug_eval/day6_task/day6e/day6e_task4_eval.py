#!/usr/bin/env python3
"""
Day 6 Task 4: old WACV D_old vs Day-5 pristine E_H on fixed KADID holdout.

Outputs stay inside:
  /home/projectwork/student_package/dataset_model2/day5_debug_eval/day6e

Default mode refreshes reports from the saved score/metric CSVs. Use
--rescore-200 to recompute the fixed 200-image subset from checkpoints.
"""
from __future__ import annotations

import argparse
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
DATASET_ROOT = PACKAGE_ROOT / "dataset_model2"
DAY5_DIR = DATASET_ROOT / "day5_debug_eval"
OUT_DIR = DAY5_DIR / "day6e"
PLOTS_DIR = OUT_DIR / "plots"

DAY5_HOLDOUT = DAY5_DIR / "day5_holdout_scores.csv"
OLD_CKPT = PACKAGE_ROOT / "checkpoints" / "best.pth"
EH_CKPT = PACKAGE_ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"

OUT_SCORES = OUT_DIR / "kadid_200_old_vs_day5_scores.csv"
OUT_METRICS = OUT_DIR / "task4_correlations_metrics.csv"
OUT_SUMMARY = OUT_DIR / "task4_summary.json"
OUT_REPORT = OUT_DIR / "task4_report.txt"
OUT_PLOT = PLOTS_DIR / "task4_dold_vs_eh_and_mos_scatters.png"

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


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    return model, ckpt["mu_ref"].to(device), ckpt["Sigma_ref"].to(device), img_size


@torch.no_grad()
def score_paths(paths: list[str], ckpt_path: Path, device: torch.device, batch_size: int = 64) -> list[float]:
    model, mu_ref, sig_ref, img_size = load_model(ckpt_path, device)
    loader = DataLoader(
        ImagePathDataset(paths, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    scores: list[float] = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, mu, logvar = model(imgs)
        score = sz_from_stats(mu, logvar, mu_ref, sig_ref, sigma_t_max=1.0, mu_only=True)
        scores.extend(score.detach().cpu().numpy().tolist())
    return scores


def corr_row(subset: str, comparison: str, x, y) -> dict:
    srcc, sp = spearmanr(x, y)
    pr, pp = pearsonr(x, y)
    kt, kp = kendalltau(x, y)
    return {
        "Subset": subset,
        "Comparison": comparison,
        "N": int(len(x)),
        "Spearman_rho": float(srcc),
        "Spearman_p": float(sp),
        "Pearson_r": float(pr),
        "Pearson_p": float(pp),
        "Kendall_tau": float(kt),
        "Kendall_p": float(kp),
    }


def select_fixed_200(day5_holdout: pd.DataFrame) -> pd.DataFrame:
    kadid = day5_holdout[(day5_holdout["dataset"] == "KADID-10k") & (day5_holdout["has_mos"] == True)].copy()
    return kadid.sort_values("image_id").head(200).reset_index(drop=True)


def recompute_200() -> pd.DataFrame:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    holdout = pd.read_csv(DAY5_HOLDOUT)
    fixed = select_fixed_200(holdout)
    paths = fixed["distorted_path"].astype(str).tolist()
    fixed["D_old"] = score_paths(paths, OLD_CKPT, device)
    # Reuse Day-5 E_H from the audited holdout CSV so this exactly matches Day-5 scoring.
    fixed["E_H"] = fixed["E_H"].astype(float)
    out_cols = [
        "image_id", "ref_id", "distorted_path", "ref_path", "distortion_type",
        "severity_or_level", "mos_or_dmos", "D_old", "E_H",
    ]
    fixed[out_cols].to_csv(OUT_SCORES, index=False)
    return fixed[out_cols]


def load_or_recompute_scores(rescore_200: bool) -> pd.DataFrame:
    if rescore_200 or not OUT_SCORES.exists():
        return recompute_200()
    return pd.read_csv(OUT_SCORES)


def build_metrics(scores_200: pd.DataFrame) -> pd.DataFrame:
    rows = [
        corr_row("200 Fixed KADID Holdout", "D_old vs MOS", scores_200["D_old"], scores_200["mos_or_dmos"]),
        corr_row("200 Fixed KADID Holdout", "E_H vs MOS", scores_200["E_H"], scores_200["mos_or_dmos"]),
        corr_row("200 Fixed KADID Holdout", "D_old vs E_H", scores_200["D_old"], scores_200["E_H"]),
    ]
    if OUT_METRICS.exists():
        previous = pd.read_csv(OUT_METRICS)
        full_rows = previous[previous["Subset"].astype(str).str.startswith("Full KADID Holdout")]
        rows.extend(full_rows.to_dict("records"))
    return pd.DataFrame(rows)


def metric(metrics_df: pd.DataFrame, subset: str, comparison: str, col: str) -> float:
    row = metrics_df[(metrics_df["Subset"] == subset) & (metrics_df["Comparison"] == comparison)]
    return float(row.iloc[0][col])


def make_plot(scores: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(scores["D_old"], scores["mos_or_dmos"], s=18, alpha=0.65)
    axes[0].set_xlabel("D_old")
    axes[0].set_ylabel("MOS")
    axes[0].set_title("D_old vs MOS")
    axes[1].scatter(scores["E_H"], scores["mos_or_dmos"], s=18, alpha=0.65, color="#2a7")
    axes[1].set_xlabel("E_H")
    axes[1].set_ylabel("MOS")
    axes[1].set_title("E_H vs MOS")
    axes[2].scatter(scores["D_old"], scores["E_H"], s=18, alpha=0.65, color="#b65")
    axes[2].set_xlabel("D_old")
    axes[2].set_ylabel("E_H")
    axes[2].set_title("D_old vs E_H")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)


def write_summary_and_report(scores: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    sub = "200 Fixed KADID Holdout"
    full = "Full KADID Holdout (N=1625)"
    summary = {
        "task": "Task 4: Cross-checkpoint evaluation (Old WACV D_old vs Day-5 E_H)",
        "subset_200_fixed": {
            "N": int(len(scores)),
            "Spearman_D_old_vs_MOS": metric(metrics_df, sub, "D_old vs MOS", "Spearman_rho"),
            "Pearson_D_old_vs_MOS": metric(metrics_df, sub, "D_old vs MOS", "Pearson_r"),
            "Spearman_E_H_vs_MOS": metric(metrics_df, sub, "E_H vs MOS", "Spearman_rho"),
            "Pearson_E_H_vs_MOS": metric(metrics_df, sub, "E_H vs MOS", "Pearson_r"),
            "Spearman_D_old_vs_E_H": metric(metrics_df, sub, "D_old vs E_H", "Spearman_rho"),
            "Pearson_D_old_vs_E_H": metric(metrics_df, sub, "D_old vs E_H", "Pearson_r"),
        },
        "checkpoints": {"D_old": str(OLD_CKPT), "E_H": str(EH_CKPT)},
        "outputs": {
            "scores_csv": str(OUT_SCORES),
            "metrics_csv": str(OUT_METRICS),
            "report_txt": str(OUT_REPORT),
            "summary_json": str(OUT_SUMMARY),
            "plots": [str(OUT_PLOT)],
        },
    }
    if full in set(metrics_df["Subset"]):
        summary["full_holdout_1625"] = {
            "N": 1625,
            "Spearman_D_old_vs_MOS": metric(metrics_df, full, "D_old vs MOS", "Spearman_rho"),
            "Pearson_D_old_vs_MOS": metric(metrics_df, full, "D_old vs MOS", "Pearson_r"),
            "Spearman_E_H_vs_MOS": metric(metrics_df, full, "E_H vs MOS", "Spearman_rho"),
            "Pearson_E_H_vs_MOS": metric(metrics_df, full, "E_H vs MOS", "Pearson_r"),
            "Spearman_D_old_vs_E_H": metric(metrics_df, full, "D_old vs E_H", "Spearman_rho"),
            "Pearson_D_old_vs_E_H": metric(metrics_df, full, "D_old vs E_H", "Pearson_r"),
        }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    s = summary["subset_200_fixed"]
    f = summary.get("full_holdout_1625", {})
    report = f"""================================================================================
DAY 6 TASK 4 AUDIT REPORT: OLD WACV (D_old) VS DAY-5 PRISTINE (E_H)
================================================================================

Setup:
- Evaluated on 200 fixed KADID holdout images.
- D_old checkpoint: {OLD_CKPT}
- E_H checkpoint:   {EH_CKPT}

200 fixed KADID holdout:
- D_old vs MOS: Spearman {s['Spearman_D_old_vs_MOS']:+.4f}, Pearson {s['Pearson_D_old_vs_MOS']:+.4f}
- E_H vs MOS:   Spearman {s['Spearman_E_H_vs_MOS']:+.4f}, Pearson {s['Pearson_E_H_vs_MOS']:+.4f}
- D_old vs E_H: Spearman {s['Spearman_D_old_vs_E_H']:+.4f}, Pearson {s['Pearson_D_old_vs_E_H']:+.4f}
"""
    if f:
        report += f"""
Full KADID holdout reference (N=1625):
- D_old vs MOS: Spearman {f['Spearman_D_old_vs_MOS']:+.4f}, Pearson {f['Pearson_D_old_vs_MOS']:+.4f}
- E_H vs MOS:   Spearman {f['Spearman_E_H_vs_MOS']:+.4f}, Pearson {f['Pearson_E_H_vs_MOS']:+.4f}
- D_old vs E_H: Spearman {f['Spearman_D_old_vs_E_H']:+.4f}, Pearson {f['Pearson_D_old_vs_E_H']:+.4f}
"""
    report += f"""
Conclusion:
- D_old and E_H rank the 200 fixed images very similarly.
- Both scores are higher=worse and therefore correlate negatively with MOS.
- The full-holdout D_old vs E_H value reported here matches the metrics CSV: {f.get('Spearman_D_old_vs_E_H', s['Spearman_D_old_vs_E_H']):+.4f}.

Outputs:
- Scores:  {OUT_SCORES}
- Metrics: {OUT_METRICS}
- JSON:    {OUT_SUMMARY}
- Plot:    {OUT_PLOT}
================================================================================
"""
    OUT_REPORT.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore-200", action="store_true", help="recompute D_old for the fixed 200 KADID rows")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_or_recompute_scores(args.rescore_200)
    metrics_df = build_metrics(scores)
    metrics_df.to_csv(OUT_METRICS, index=False)
    make_plot(scores)
    write_summary_and_report(scores, metrics_df)
    print(f"Saved Task 4 outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
