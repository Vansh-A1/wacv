#!/usr/bin/env python3
"""Negate PSNR/SSIM, recompute WACV correlations, and plot scatter graphs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
METRICS = ["NIQE", "LR_content", "PSNR_neg", "SSIM_neg", "LPIPS", "DISTS"]
INPUTS = {
    "SwinSR": HERE / "task_e_swinsr_metrics.csv",
    "HAT": HERE / "task_e_hat_metrics.csv",
}


def load_and_adjust(path):
    df = pd.read_csv(path)
    df["PSNR_original"] = df["PSNR"]
    df["SSIM_original"] = df["SSIM"]
    df["PSNR_neg"] = -df["PSNR"]
    df["SSIM_neg"] = -df["SSIM"]
    df["PSNR"] = df["PSNR_neg"]
    df["SSIM"] = df["SSIM_neg"]
    return df


def correlation_rows(df, variant):
    rows = []
    for method in ["SwinSR", "HAT", "pooled"]:
        subset = df if method == "pooled" else df[df["method"] == method]
        for metric in METRICS:
            valid = subset[["D_WACV", metric]].dropna()
            rho, p_value = spearmanr(valid["D_WACV"], valid[metric])
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "pair": f"D_WACV vs {metric}",
                    "Spearman_rho": float(rho),
                    "p_value": float(p_value),
                    "N": int(len(valid)),
                }
            )
    return rows


def plot_metric(df, metric, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=160)
    colors = {"SwinSR": "#1f77b4", "HAT": "#d62728"}
    for method, part in df.groupby("method"):
        ax.scatter(
            part["D_WACV"],
            part[metric],
            s=22,
            alpha=0.78,
            label=method,
            color=colors.get(method),
            edgecolors="none",
        )
    valid = df[["D_WACV", metric]].dropna()
    rho, p_value = spearmanr(valid["D_WACV"], valid[metric])
    ax.set_xlabel("D_WACV (higher = worse)")
    ax.set_ylabel(metric)
    ax.set_title(f"D_WACV vs {metric}")
    ax.text(
        0.02,
        0.98,
        f"Spearman rho={rho:.4f}\np={p_value:.3g}\nN={len(valid)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    plots_dir = HERE / "plots"
    plots_dir.mkdir(exist_ok=True)

    adjusted = []
    for method, path in INPUTS.items():
        df = load_and_adjust(path)
        adjusted.append(df)
        df.to_csv(HERE / f"task_e_{method.lower()}_metrics_psnr_ssim_negated.csv", index=False)

    pooled = pd.concat(adjusted, ignore_index=True)
    pooled.to_csv(HERE / "task_e_pooled_metrics_psnr_ssim_negated.csv", index=False)

    corr = pd.DataFrame(correlation_rows(pooled, "PSNR_and_SSIM_negated"))
    corr.to_csv(HERE / "task_e_correlations_psnr_ssim_negated.csv", index=False)

    for metric in METRICS:
        plot_metric(pooled, metric, plots_dir / f"wacv_vs_{metric}.png")

    lines = [
        "Task E copy with PSNR and SSIM negated",
        "Original CSVs were copied from day2_tasks/task_e_metric_correspondence.",
        "PSNR_neg = -PSNR_original; SSIM_neg = -SSIM_original.",
        "In the adjusted CSVs, columns PSNR and SSIM are replaced with the negated values.",
        "",
        "Pooled correlations:",
    ]
    pooled_corr = corr[corr["method"] == "pooled"]
    for _, row in pooled_corr.iterrows():
        lines.append(
            f"{row['pair']}: rho={row['Spearman_rho']:.6f}, p={row['p_value']:.6g}, N={int(row['N'])}"
        )
    lines.extend(
        [
            "",
            f"Adjusted pooled CSV: {HERE / 'task_e_pooled_metrics_psnr_ssim_negated.csv'}",
            f"Correlation CSV: {HERE / 'task_e_correlations_psnr_ssim_negated.csv'}",
            f"Plots folder: {plots_dir}",
        ]
    )
    (HERE / "task_e_negated_psnr_ssim_summary.txt").write_text("\n".join(lines) + "\n")
    print((HERE / "task_e_negated_psnr_ssim_summary.txt").read_text())


if __name__ == "__main__":
    main()
