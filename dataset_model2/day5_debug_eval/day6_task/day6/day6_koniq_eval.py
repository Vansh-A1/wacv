#!/usr/bin/env python3
"""
Day 6 – KonIQ-10k MOS Integration & Correlation Analysis
==========================================================
Task 0:
  - Load official KonIQ-10k MOS file (koniq10k_scores.csv)
  - Join onto day5_holdout_scores.csv by image_id / filename
  - Set has_mos=True and fill mos_or_dmos for KonIQ holdout rows
  - Recompute Spearman for E_H, E_A, Q on KonIQ holdout only
  - Save day5_correlations_koniq.csv + 3 scatter plots

Inputs:
  - day5_holdout_scores.csv  (Day-5 scored holdout set)
  - diagnostics.json         (Day-5 diagnostics)
  - ea_reference_set_diagnostics.csv (EA ref paths)
  - ref_splits_seed42/       (split CSVs)

Output root: day6/
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_ROOT   = Path("/home/projectwork/student_package")
DATASET_ROOT   = PACKAGE_ROOT / "dataset_model2"
DAY5_DIR       = DATASET_ROOT / "day5_debug_eval"
DAY6_DIR       = DAY5_DIR / "day6"

DAY5_HOLDOUT   = DAY5_DIR / "day5_holdout_scores.csv"
DIAGNOSTICS    = DAY5_DIR / "diagnostics.json"
KONIQ_SCORES   = DAY6_DIR / "koniq10k_scores.csv"

OUT_CORR_CSV   = DAY6_DIR / "day5_correlations_koniq.csv"
OUT_UNMATCHED  = DAY6_DIR / "koniq_unmatched_holdout_rows.csv"
OUT_PLOTS_DIR  = DAY6_DIR / "plots"
OUT_SUMMARY    = DAY6_DIR / "day6_summary.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def corr_pair(x, y):
    """Compute Spearman, Pearson, Kendall for a pair of arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return {
            "N": n,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "pearson_r":    np.nan, "pearson_p":  np.nan,
            "kendall_tau":  np.nan, "kendall_p":  np.nan,
        }
    sp = spearmanr(x[mask], y[mask])
    pr = pearsonr(x[mask], y[mask])
    kt = kendalltau(x[mask], y[mask])
    return {
        "N":            n,
        "spearman_rho": float(sp.statistic),
        "spearman_p":   float(sp.pvalue),
        "pearson_r":    float(pr.statistic),
        "pearson_p":    float(pr.pvalue),
        "kendall_tau":  float(kt.statistic),
        "kendall_p":    float(kt.pvalue),
    }


def compute_correlations_koniq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman / Pearson / Kendall for E_H, E_A, Q vs KonIQ MOS
    on the KonIQ holdout subset only.
    """
    mos_df = df[df["has_mos"] == True].copy()
    mos_df["mos_float"] = pd.to_numeric(mos_df["mos_or_dmos"], errors="coerce")
    mos_df = mos_df.dropna(subset=["mos_float"])

    rows = []
    for score_name in ["E_H", "E_A", "Q"]:
        c = corr_pair(mos_df[score_name], mos_df["mos_float"])
        rows.append({
            "dataset": "KONIQ-10k",
            "target":  "MOS",
            "score":   score_name,
            **c,
        })
    return pd.DataFrame(rows)


def plot_scatter_koniq(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """
    3 scatter plots as specified:
      1. E_H  vs KonIQ MOS  -> KonIQ_E_H_vs_mos.png
      2. E_A  vs KonIQ MOS  -> KonIQ_E_A_vs_mos.png
      3. Q    vs KonIQ MOS  -> KonIQ_Q_vs_mos.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mos_df = df[df["has_mos"] == True].copy()
    mos_df["mos_float"] = pd.to_numeric(mos_df["mos_or_dmos"], errors="coerce")

    plot_specs = [
        ("E_H", "KonIQ MOS", "KonIQ_E_H_vs_mos.png"),
        ("E_A", "KonIQ MOS", "KonIQ_E_A_vs_mos.png"),
        ("Q",   "KonIQ MOS", "KonIQ_Q_vs_mos.png"),
    ]

    made = []
    for score_name, ylabel, fname in plot_specs:
        valid = mos_df[[score_name, "mos_float"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(valid) < 3:
            print(f"  [SKIP] {fname}: only {len(valid)} valid points")
            continue

        sp = spearmanr(valid[score_name], valid["mos_float"])

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(valid[score_name], valid["mos_float"], s=12, alpha=0.55,
                   color="#2196F3", edgecolors="none")
        ax.set_xlabel(score_name, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"KonIQ-10k Holdout: {score_name} vs KonIQ MOS\n"
            f"Spearman ρ = {sp.statistic:.4f}  (N={len(valid)})",
            fontsize=11,
        )
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(str(path))
        print(f"  Saved: {path}")

    return made


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DAY6_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Day 6 – KonIQ MOS Integration & Correlation Analysis")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load day-5 holdout scores
    # ------------------------------------------------------------------
    print("\n[1] Loading Day-5 holdout scores …")
    scores = pd.read_csv(DAY5_HOLDOUT, dtype=str, keep_default_na=False)
    print(f"    Total rows: {len(scores)}")
    koniq_mask = scores["dataset"] == "KONIQ-10k"
    print(f"    KonIQ-10k rows: {koniq_mask.sum()}")

    # ------------------------------------------------------------------
    # 2. Load official KonIQ MOS file
    # ------------------------------------------------------------------
    print("\n[2] Loading official KonIQ-10k MOS file …")
    if not KONIQ_SCORES.is_file():
        raise FileNotFoundError(
            f"koniq10k_scores.csv not found at {KONIQ_SCORES}.\n"
            "Download from: http://datasets.vqa.mmsp-kn.de/archives/"
            "koniq10k_scores_and_distributions.zip"
        )
    mos_df = pd.read_csv(KONIQ_SCORES)
    print(f"    Columns: {list(mos_df.columns)}")
    print(f"    Shape:   {mos_df.shape}")

    # Build lookup: image_id (numeric stem) -> MOS
    # image_name is like "10004473376.jpg"; strip extension for the key
    mos_df["image_id_str"] = mos_df["image_name"].map(
        lambda n: Path(str(n)).stem
    )
    mos_lookup: dict[str, float] = dict(
        zip(mos_df["image_id_str"], mos_df["MOS"])
    )
    print(f"    MOS lookup entries: {len(mos_lookup)}")

    # ------------------------------------------------------------------
    # 3. Join: fill has_mos and mos_or_dmos for KonIQ rows
    # ------------------------------------------------------------------
    print("\n[3] Joining MOS onto KonIQ holdout rows …")
    scores = scores.copy()

    # Ensure correct dtypes for mutable columns
    scores["has_mos"]     = scores["has_mos"].map(
        lambda v: True if str(v).strip().lower() == "true" else False
    )
    scores["mos_or_dmos"] = scores["mos_or_dmos"].astype(object)

    matched = 0
    unmatched_rows = []
    for idx, row in scores[koniq_mask].iterrows():
        img_id = str(row["image_id"]).strip()
        if img_id in mos_lookup:
            scores.at[idx, "has_mos"]     = True
            scores.at[idx, "mos_or_dmos"] = float(mos_lookup[img_id])
            matched += 1
        else:
            unmatched_rows.append(row.to_dict())

    print(f"    Matched and filled: {matched} / {koniq_mask.sum()} KonIQ rows")
    print(f"    has_mos=True for KonIQ: {(scores[koniq_mask]['has_mos'] == True).sum()}")
    pd.DataFrame(unmatched_rows).to_csv(OUT_UNMATCHED, index=False)
    print(f"    Unmatched KonIQ rows saved: {len(unmatched_rows)} -> {OUT_UNMATCHED}")

    # ------------------------------------------------------------------
    # 4. Recompute Spearman / Pearson / Kendall for KonIQ holdout
    # ------------------------------------------------------------------
    print("\n[4] Computing correlations on KonIQ holdout with MOS …")

    koniq_scored = scores[koniq_mask].copy()
    # Numeric conversion for score columns
    for col in ["E_H", "E_A", "Q"]:
        koniq_scored[col] = pd.to_numeric(koniq_scored[col], errors="coerce")
    koniq_scored["mos_or_dmos"] = pd.to_numeric(
        koniq_scored["mos_or_dmos"], errors="coerce"
    )

    corr_df = compute_correlations_koniq(koniq_scored)

    # Pretty-print Spearman
    mos_corr = corr_df[corr_df["target"] == "MOS"]
    for _, row in mos_corr.iterrows():
        print(f"    {row['score']:4s}  Spearman ρ = {row['spearman_rho']:+.4f}  "
              f"(N={int(row['N'])})")

    # ------------------------------------------------------------------
    # 5. Save correlation CSV
    # ------------------------------------------------------------------
    corr_df.to_csv(OUT_CORR_CSV, index=False)
    print(f"\n[5] Saved correlations → {OUT_CORR_CSV}")

    # ------------------------------------------------------------------
    # 6. Save 3 scatter plots
    # ------------------------------------------------------------------
    print("\n[6] Generating 3 scatter plots …")
    plots = plot_scatter_koniq(koniq_scored, OUT_PLOTS_DIR)

    # ------------------------------------------------------------------
    # 7. Write summary JSON
    # ------------------------------------------------------------------
    srcc = {
        r["score"]: float(r["spearman_rho"])
        for _, r in mos_corr.iterrows()
    }
    summary = {
        "day6_task": "KonIQ-10k MOS integration",
        "koniq_holdout_rows": int(koniq_mask.sum()),
        "koniq_mos_matched": matched,
        "koniq_mos_unmatched": int(koniq_mask.sum()) - matched,
        "koniq_mos_file": str(KONIQ_SCORES),
        "spearman_koniq_mos": srcc,
        "outputs": {
            "correlations_csv":  str(OUT_CORR_CSV),
            "unmatched_rows_csv": str(OUT_UNMATCHED),
            "scatter_plots":     plots,
            "summary_json":      str(OUT_SUMMARY),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[7] Summary → {OUT_SUMMARY}")

    print("\n" + "=" * 65)
    print("Day 6 complete.")
    print("=" * 65)
    print("\nOutputs:")
    print(f"  Correlations CSV : {OUT_CORR_CSV}")
    print(f"  Plots dir        : {OUT_PLOTS_DIR}")
    for p in plots:
        print(f"    {p}")
    print(f"  Summary JSON     : {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
