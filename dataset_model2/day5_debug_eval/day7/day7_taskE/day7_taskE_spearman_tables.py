#!/usr/bin/env python3
"""Day 7 Task E: Spearman/Pearson tables using Task D scores."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path("/home/projectwork/student_package")
DAY5_ROOT = ROOT / "dataset_model2" / "day5_debug_eval"
DAY7_ROOT = DAY5_ROOT / "day7"
OUT_DIR = DAY7_ROOT / "day7_taskE"

TASKD_DIR = DAY7_ROOT / "day7_taskD"
TASKC_DIR = DAY7_ROOT / "day7_taskc"
TASKA_DIR = DAY7_ROOT / "day7_taskA"
DAY6_DIR = DAY5_ROOT / "day6_task"

TASKD_HOLDOUT = TASKD_DIR / "taskD_holdout_scores.csv"
TASKD_VAL = TASKD_DIR / "taskD_val_scores.csv"
TASKC_SELECTION = TASKC_DIR / "taskc_checkpoint_selection_table.csv"
DAY7_REF_COMPARE = TASKA_DIR / "day7_reference_stats_comparison.csv"

DAY5_HOLDOUT = DAY5_ROOT / "day5_holdout_scores.csv"
DAY6_REF_COMPARE = DAY6_DIR / "day6_b" / "task1_similarity_metrics.csv"
DAY6_QZ_SUMMARY = DAY6_DIR / "day6d" / "task3_qz_summary_table.csv"

DATASET_LABELS = {
    "KADID-10k": "KADID",
    "TID2013": "TID",
    "KONIQ-10k": "KonIQ",
}

SCORE_LABELS = {
    "E_H": "E_H",
    "E_A": "E_A",
    "Q_raw": "Q_raw",
    "Qz_AmH": "Q_z^(A-H)",
    "Qz_HmA": "Q_z^(H-A)",
}

WITHIN_TYPES = [
    "Gaussian blur",
    "Lens blur",
    "High sharpen",
    "Pixelate",
    "JPEG",
    "White noise",
]


def finite_xy(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def corr_values(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float | int]:
    x, y = finite_xy(df, x_col, y_col)
    n = int(len(x))
    if n < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {
            "N": n,
            "SRCC": math.nan,
            "SRCC_p": math.nan,
            "Pearson": math.nan,
            "Pearson_p": math.nan,
        }
    srcc = spearmanr(x, y)
    pr = pearsonr(x, y)
    return {
        "N": n,
        "SRCC": float(srcc.statistic),
        "SRCC_p": float(srcc.pvalue),
        "Pearson": float(pr.statistic),
        "Pearson_p": float(pr.pvalue),
    }


def require_paths() -> None:
    required = [
        TASKD_HOLDOUT,
        TASKD_VAL,
        TASKC_SELECTION,
        DAY7_REF_COMPARE,
        DAY5_HOLDOUT,
        DAY6_REF_COMPARE,
        DAY6_QZ_SUMMARY,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def e1_mos_tables(holdout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, dataset_label in DATASET_LABELS.items():
        subset = holdout[holdout["dataset"] == dataset]
        for score_col, score_label in SCORE_LABELS.items():
            c = corr_values(subset, score_col, "mos_or_dmos")
            rows.append(
                {
                    "Dataset": dataset_label,
                    "Score": score_label,
                    "N": c["N"],
                    "SRCC vs MOS": c["SRCC"],
                    "SRCC p": c["SRCC_p"],
                    "Pearson vs MOS": c["Pearson"],
                    "Pearson p": c["Pearson_p"],
                }
            )
    return pd.DataFrame(rows)


def e2_overall_severity(holdout: pd.DataFrame) -> pd.DataFrame:
    severity_df = holdout[
        holdout["dataset"].isin(["KADID-10k", "TID2013"])
        & pd.to_numeric(holdout["severity_or_level"], errors="coerce").notna()
    ].copy()
    rows: list[dict[str, object]] = []
    for score_col, score_label in [
        ("E_H", "E_H"),
        ("E_A", "E_A"),
        ("Qz_HmA", "Q_z^(H-A)"),
    ]:
        c = corr_values(severity_df, score_col, "severity_or_level")
        rows.append(
            {
                "Score": score_label,
                "Subset": "KADID+TID holdout, all distortion types pooled",
                "N": c["N"],
                "SRCC(score,severity)": c["SRCC"],
                "SRCC p": c["SRCC_p"],
            }
        )
    return pd.DataFrame(rows)


def e3_within_type_severity(holdout: pd.DataFrame) -> pd.DataFrame:
    kadid = holdout[holdout["dataset"] == "KADID-10k"].copy()
    rows: list[dict[str, object]] = []
    for distortion_type in WITHIN_TYPES:
        subset = kadid[kadid["distortion_type"] == distortion_type]
        eh = corr_values(subset, "E_H", "severity_or_level")
        ea = corr_values(subset, "E_A", "severity_or_level")
        qz = corr_values(subset, "Qz_HmA", "severity_or_level")
        rows.append(
            {
                "Type": distortion_type,
                "SRCC(E_H,sev)": eh["SRCC"],
                "SRCC(E_A,sev)": ea["SRCC"],
                "SRCC(Q_z^(H-A),sev)": qz["SRCC"],
                "N": eh["N"],
            }
        )
    return pd.DataFrame(rows)


def read_day6_qz(metric: str, dataset: str = "KADID holdout") -> float:
    df = pd.read_csv(DAY6_QZ_SUMMARY)
    row = df[df["Dataset"] == dataset]
    if row.empty:
        return math.nan
    return float(row.iloc[0][metric])


def read_day6_cosine() -> float:
    df = pd.read_csv(DAY6_REF_COMPARE)
    row = df[df["vector_name"] == "mu_ref"]
    return float(row.iloc[0]["cosine_similarity"]) if not row.empty else math.nan


def read_day7_cosine() -> float:
    df = pd.read_csv(DAY7_REF_COMPARE)
    row = df[(df["vector"] == "mu_ref") & (df["right"] == "pristine_E_H_ref")]
    return float(row.iloc[0]["cosine_similarity"]) if not row.empty else math.nan


def e4_day6_vs_day7(holdout: pd.DataFrame, day5_holdout: pd.DataFrame) -> pd.DataFrame:
    old_kadid = day5_holdout[day5_holdout["dataset"] == "KADID-10k"].copy()
    new_kadid = holdout[holdout["dataset"] == "KADID-10k"].copy()

    old_ea_mos = corr_values(old_kadid, "E_A", "mos_or_dmos")["SRCC"]
    old_eh_mos = corr_values(old_kadid, "E_H", "mos_or_dmos")["SRCC"]
    old_qz_mos = read_day6_qz("SRCC Q_z^{(A-H)}")

    new_ea_mos = corr_values(new_kadid, "E_A", "mos_or_dmos")["SRCC"]
    new_eh_mos = corr_values(new_kadid, "E_H", "mos_or_dmos")["SRCC"]
    new_qz_mos = corr_values(new_kadid, "Qz_AmH", "mos_or_dmos")["SRCC"]

    old_blur = corr_values(old_kadid[old_kadid["distortion_type"] == "Gaussian blur"], "E_A", "severity_or_level")["SRCC"]
    new_blur = corr_values(new_kadid[new_kadid["distortion_type"] == "Gaussian blur"], "E_A", "severity_or_level")["SRCC"]
    old_sharpen = corr_values(old_kadid[old_kadid["distortion_type"] == "High sharpen"], "E_A", "severity_or_level")["SRCC"]
    new_sharpen = corr_values(new_kadid[new_kadid["distortion_type"] == "High sharpen"], "E_A", "severity_or_level")["SRCC"]

    return pd.DataFrame(
        [
            {
                "Metric": "cosine(mu_ref_H, mu_ref_A)",
                "Day-6 (old E_A_ref)": read_day6_cosine(),
                "Day-7 (this run)": read_day7_cosine(),
                "Notes": "mu_ref cosine similarity",
            },
            {
                "Metric": "KADID SRCC E_A vs MOS",
                "Day-6 (old E_A_ref)": old_ea_mos,
                "Day-7 (this run)": new_ea_mos,
                "Notes": "KADID holdout only",
            },
            {
                "Metric": "KADID SRCC Qz_AmH vs MOS",
                "Day-6 (old E_A_ref)": old_qz_mos,
                "Day-7 (this run)": new_qz_mos,
                "Notes": "MOS-oriented Q_z^(A-H); higher MOS is better in KADID",
            },
            {
                "Metric": "KADID SRCC E_H vs MOS",
                "Day-6 (old E_A_ref)": old_eh_mos,
                "Day-7 (this run)": new_eh_mos,
                "Notes": "Frozen pristine scorer, should match aside from scoring precision",
            },
            {
                "Metric": "Blur E_A vs severity",
                "Day-6 (old E_A_ref)": old_blur,
                "Day-7 (this run)": new_blur,
                "Notes": "KADID Gaussian blur holdout",
            },
            {
                "Metric": "Sharpen E_A vs severity",
                "Day-6 (old E_A_ref)": old_sharpen,
                "Day-7 (this run)": new_sharpen,
                "Notes": "KADID High sharpen holdout",
            },
        ]
    )


def write_report(
    e1: pd.DataFrame,
    e2: pd.DataFrame,
    e3: pd.DataFrame,
    e4: pd.DataFrame,
    selected_ckpt: str,
    summary: dict[str, object],
) -> None:
    report = [
        "Day 7 Task E - Spearman tables",
        "",
        "Inputs:",
        f"Task D holdout scores: {TASKD_HOLDOUT}",
        f"Task D val scores: {TASKD_VAL}",
        f"Selected Day 7 E_A checkpoint: {selected_ckpt}",
        f"Day 7 new reference comparison: {DAY7_REF_COMPARE}",
        f"Day 6 baseline folder: {DAY6_DIR}",
        "",
        "Conventions:",
        "E_H = frozen pristine energy.",
        "E_A = artifact energy from the selected Day 7 checkpoint and new artifact reference.",
        "Q_raw = E_A - E_H.",
        "Q_z^(A-H) = z_A - z_H; Q_z^(H-A) = z_H - z_A.",
        "For severity tables, higher severity means stronger distortion; Q_z^(H-A) is the main severity-oriented Q.",
        "For MOS comparison in E4, Q_z^(A-H) is reported because KADID MOS is higher-is-better.",
        "",
        "E1 - MOS correlations:",
        e1.to_string(index=False),
        "",
        "E2 - Overall severity correlations:",
        e2.to_string(index=False),
        "",
        "E3 - Within-type KADID severity correlations:",
        e3.to_string(index=False),
        "",
        "E4 - Day 6 vs Day 7 comparison:",
        e4.to_string(index=False),
        "",
        "Summary:",
        json.dumps(summary, indent=2),
        "",
    ]
    (OUT_DIR / "taskE_report.txt").write_text("\n".join(report))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    require_paths()

    holdout = pd.read_csv(TASKD_HOLDOUT)
    val = pd.read_csv(TASKD_VAL)
    day5_holdout = pd.read_csv(DAY5_HOLDOUT)
    selection = pd.read_csv(TASKC_SELECTION)
    selected = selection[selection["selected"].astype(str).str.lower().eq("yes")]
    selected_ckpt = str(selected.iloc[0]["ckpt path"]) if not selected.empty else "UNKNOWN"

    e1 = e1_mos_tables(holdout)
    e2 = e2_overall_severity(holdout)
    e3 = e3_within_type_severity(holdout)
    e4 = e4_day6_vs_day7(holdout, day5_holdout)

    e1_path = OUT_DIR / "taskE_E1_mos_correlations.csv"
    e2_path = OUT_DIR / "taskE_E2_overall_severity_correlations.csv"
    e3_path = OUT_DIR / "taskE_E3_within_type_severity_correlations.csv"
    e4_path = OUT_DIR / "taskE_E4_day6_vs_day7_comparison.csv"
    e1.to_csv(e1_path, index=False)
    e2.to_csv(e2_path, index=False)
    e3.to_csv(e3_path, index=False)
    e4.to_csv(e4_path, index=False)

    summary = {
        "output_dir": str(OUT_DIR),
        "taskD_holdout_scores": str(TASKD_HOLDOUT),
        "taskD_val_scores": str(TASKD_VAL),
        "day6_baseline_dir": str(DAY6_DIR),
        "selected_day7_ea_checkpoint": selected_ckpt,
        "holdout_rows": int(len(holdout)),
        "val_rows": int(len(val)),
        "holdout_dataset_counts": {str(k): int(v) for k, v in holdout["dataset"].value_counts().to_dict().items()},
        "val_dataset_counts": {str(k): int(v) for k, v in val["dataset"].value_counts().to_dict().items()},
        "generated_files": [
            str(e1_path),
            str(e2_path),
            str(e3_path),
            str(e4_path),
            str(OUT_DIR / "taskE_summary.json"),
            str(OUT_DIR / "taskE_report.txt"),
        ],
    }
    (OUT_DIR / "taskE_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(e1, e2, e3, e4, selected_ckpt, summary)

    print(f"Task E complete. Outputs saved to: {OUT_DIR}")
    print(f"Selected checkpoint: {selected_ckpt}")
    print("Generated:")
    for path in summary["generated_files"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
