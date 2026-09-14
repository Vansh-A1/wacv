#!/usr/bin/env python3
"""Task3 ablation: Qz from ranked E_A and ranked E_H.

This script does not rescore images. It reuses Day11/Task12 validation and
holdout CSV outputs, recomputes 11b/Task12 selection with a positive E_H
severity reward, and writes a joined holdout CSV.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/projectwork/student_package/day11_day12")
OUT = ROOT / "task3"
DAY11_VAL_SELECTION = ROOT / "val_selection_table.csv"
DAY11_HOLDOUT = ROOT / "holdout_rankall_scores.csv"
TASK12_ROOT = ROOT / "task12" / "day11b_pristine_rank"
TASK12_VAL_SELECTION = TASK12_ROOT / "val_selection_table.csv"
TASK12_HOLDOUT = TASK12_ROOT / "holdout_metrics.csv"
KADID = "KADID-10k"


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


def selected_epoch(table: pd.DataFrame, lambda_rank: float) -> int:
    sub = table[(table["lambda_rank"].astype(float) == float(lambda_rank)) & table["selected"].astype(str).eq("yes")]
    if len(sub) != 1:
        raise ValueError(f"Expected one selected row for lambda={lambda_rank}, got {len(sub)}")
    return int(sub.iloc[0]["epoch"])


def recompute_positive_eh_selection() -> pd.DataFrame:
    src = pd.read_csv(TASK12_VAL_SELECTION)
    out = src.copy()
    # 11b note: selection should reward positive E_H severity agreement, not use
    # the Day11 E_A sign formula where blur/lens are rewarded as negative.
    out["positive_EH_selection_score"] = (
        out["blur_SRCC"]
        + out["lens_SRCC"]
        + out["sharpen_SRCC"]
        + out["pixelate_SRCC"]
    )
    out["positive_EH_selected"] = "no"
    for lam, group in out.groupby("lambda_rank"):
        out.loc[group["positive_EH_selection_score"].idxmax(), "positive_EH_selected"] = "yes"
    out.to_csv(OUT / "task3_11b_positive_EH_selection_table.csv", index=False)
    return out


def z_from_val_and_holdout(val_scores: pd.DataFrame, holdout_values: pd.Series, score_col: str) -> tuple[pd.Series, dict]:
    kadid_val = val_scores[val_scores["dataset"].eq(KADID)]
    mean = float(kadid_val[score_col].mean())
    std = float(kadid_val[score_col].std(ddof=0))
    z = (pd.to_numeric(holdout_values, errors="coerce") - mean) / std
    return z, {"mean": mean, "std": std, "N": int(len(kadid_val)), "score_col": score_col}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    day11_sel = pd.read_csv(DAY11_VAL_SELECTION)
    task12_pos_sel = recompute_positive_eh_selection()
    day11_hold = pd.read_csv(DAY11_HOLDOUT)
    task12_hold = pd.read_csv(TASK12_HOLDOUT)

    # Main Day11 rank-all E_A selection remains the Day11 task rule.
    ea_lambda = 0.1 if selected_epoch(day11_sel, 0.1) == 5 else 0.1
    ea_epoch = selected_epoch(day11_sel, ea_lambda)

    # 11b/Task12 selection is recomputed here with positive E_H severity reward.
    pos_rows = task12_pos_sel[task12_pos_sel["positive_EH_selected"].eq("yes")]
    pos_rows = pos_rows.sort_values("positive_EH_selection_score", ascending=False)
    eh_lambda = float(pos_rows.iloc[0]["lambda_rank"])
    eh_epoch = int(pos_rows.iloc[0]["epoch"])

    ranked_ea_col = f"rankall_E_A_lambda_{ea_lambda}"
    ranked_ea_z_col = f"rankall_z_A_lambda_{ea_lambda}"
    ranked_ea_qz_col = f"rankall_Qz_AmH_lambda_{ea_lambda}"
    ranked_eh_col = f"ranked_E_H_lambda_{eh_lambda}"

    hold = day11_hold.merge(
        task12_hold[["image_id", ranked_eh_col]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    val_eh = pd.read_csv(TASK12_ROOT / f"lambda_{eh_lambda}" / f"val_scores_epoch_{eh_epoch:04d}.csv")
    hold["ranked_z_H"], eh_norm = z_from_val_and_holdout(val_eh, hold[ranked_eh_col], "E_H_ranked")
    hold["ablation2_Qz_rankedEA_minus_rankedEH"] = hold[ranked_ea_z_col] - hold["ranked_z_H"]
    hold["ablation2_Qz_rankedEH_minus_rankedEA"] = hold["ranked_z_H"] - hold[ranked_ea_z_col]

    wanted_cols = [
        "image_id",
        "distortion_type",
        "severity_or_level",
        "mos_or_dmos",
        "E_H",
        ranked_eh_col,
        "S0_Qz_AmH_ep05",
        ranked_ea_col,
        ranked_ea_qz_col,
        "ranked_z_H",
        ranked_ea_z_col,
        "ablation2_Qz_rankedEA_minus_rankedEH",
        "ablation2_Qz_rankedEH_minus_rankedEA",
    ]
    hold[wanted_cols].to_csv(OUT / "task3_wanted_EH_rankedEH_S0Qz_rankallQz_ablation2.csv", index=False)

    metric_rows = []
    metrics = [
        ("wanted_EH_unranked", "E_H", "mos_or_dmos", None),
        ("ranked_EH_positive_selected", ranked_eh_col, "mos_or_dmos", None),
        ("S0_Qz_AmH_ep05", "S0_Qz_AmH_ep05", "mos_or_dmos", None),
        ("rankall_Qz_EA_vs_unrankedEH", ranked_ea_qz_col, "mos_or_dmos", None),
        ("ablation2_Qz_rankedEA_minus_rankedEH", "ablation2_Qz_rankedEA_minus_rankedEH", "mos_or_dmos", None),
        ("ablation2_Qz_rankedEH_minus_rankedEA", "ablation2_Qz_rankedEH_minus_rankedEA", "mos_or_dmos", None),
        ("ranked_EA_blur_severity", ranked_ea_col, "severity_or_level", "Gaussian blur"),
        ("ranked_EH_blur_severity", ranked_eh_col, "severity_or_level", "Gaussian blur"),
        ("ranked_EA_sharpen_severity", ranked_ea_col, "severity_or_level", "High sharpen"),
        ("ranked_EH_sharpen_severity", ranked_eh_col, "severity_or_level", "High sharpen"),
        ("ranked_EA_pixelate_severity", ranked_ea_col, "severity_or_level", "Pixelate"),
        ("ranked_EH_pixelate_severity", ranked_eh_col, "severity_or_level", "Pixelate"),
    ]
    for name, col, target, dtype in metrics:
        sub = hold if dtype is None else hold[hold["distortion_type"].eq(dtype)]
        sr, sp, n, pr, pp = finite_corr(sub[col], sub[target])
        metric_rows.append(
            {
                "metric": name,
                "score_col": col,
                "target": target,
                "subset": "KADID holdout" if dtype is None else f"KADID holdout {dtype}",
                "N": n,
                "spearman_rho": sr,
                "spearman_p": sp,
                "pearson_r": pr,
                "pearson_p": pp,
            }
        )
    corr = pd.DataFrame(metric_rows)
    corr.to_csv(OUT / "task3_correlations.csv", index=False)

    summary = {
        "task": "Ablation2 Qz from ranked E_A and ranked E_H",
        "output_dir": str(OUT),
        "inputs": {
            "day11_val_selection": str(DAY11_VAL_SELECTION),
            "day11_holdout": str(DAY11_HOLDOUT),
            "task12_val_selection": str(TASK12_VAL_SELECTION),
            "task12_holdout": str(TASK12_HOLDOUT),
        },
        "selection": {
            "ranked_EA_source": "Day11 selection table",
            "ranked_EA_lambda": ea_lambda,
            "ranked_EA_epoch": ea_epoch,
            "ranked_EH_source": "Task12/11b positive EH severity selection recomputed in task3",
            "ranked_EH_lambda": eh_lambda,
            "ranked_EH_epoch": eh_epoch,
            "ranked_EH_normalization": eh_norm,
        },
        "outputs": [
            "task3_11b_positive_EH_selection_table.csv",
            "task3_wanted_EH_rankedEH_S0Qz_rankallQz_ablation2.csv",
            "task3_correlations.csv",
            "task3_report.txt",
            "task3_summary.json",
        ],
    }
    (OUT / "task3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "Task3 ablation report",
        "",
        "Interpretation of supervisor note:",
        "11b/Task12 checkpoint selection should reward positive E_H severity correlations.",
        "Ablation2 Qz uses ranked E_A and ranked E_H together.",
        "",
        f"Selected ranked E_A: lambda={ea_lambda}, epoch={ea_epoch}",
        f"Selected ranked E_H by positive-EH rule: lambda={eh_lambda}, epoch={eh_epoch}",
        "",
        "Key correlations:",
        corr.to_string(index=False),
        "",
        "Main CSV:",
        str(OUT / "task3_wanted_EH_rankedEH_S0Qz_rankallQz_ablation2.csv"),
    ]
    (OUT / "task3_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
