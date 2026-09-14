#!/usr/bin/env python3
"""Correct Task3 stage0 ranked-EA/ranked-EH baseline.

This fixes the old Task3 mistake: ranked_E_H_lambda_0.1 from the Day-11b
holdout file was tied to the previously selected epoch 5, while the later
positive-E_H rule selected epoch 20. This script selects ranked-E_H from
validation, then scores KADID holdout with the exact selected checkpoint.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr


ROOT = Path("/home/projectwork/student_package")
DAY_ROOT = ROOT / "day11_day12"
TASK12_ROOT = DAY_ROOT / "task12" / "day11b_pristine_rank"
OUT_ROOT = DAY_ROOT / "task3_correction" / "stage0_corrected_baseline"
KADID = "KADID-10k"
LOCKED_TYPES = {
    "blur": "Gaussian blur",
    "lens": "Lens blur",
    "sharpen": "High sharpen",
    "pixelate": "Pixelate",
}

sys.path.insert(0, str(DAY_ROOT))
import day11_rankall_pipeline as day11  # noqa: E402

sys.path.insert(0, str(TASK12_ROOT.parent))
import day11b_pristine_rank_pipeline as task12  # noqa: E402


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


def stored_epoch(ckpt_path: Path, device: torch.device) -> int | None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ep = ckpt.get("epoch")
    return None if ep is None else int(ep)


def z_stats(series: pd.Series) -> dict[str, float]:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    return {
        "mean": float(vals.mean()),
        "std_population": float(vals.std(ddof=0)),
        "N": int(len(vals)),
    }


def add_z(series: pd.Series, stats: dict[str, float]) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") - stats["mean"]) / stats["std_population"]


def build_positive_eh_selection(device: torch.device) -> tuple[pd.DataFrame, dict]:
    rows = []
    for lam_dir in sorted(TASK12_ROOT.glob("lambda_*")):
        if not lam_dir.is_dir():
            continue
        lam = float(lam_dir.name.split("_", 1)[1])
        for val_path in sorted(lam_dir.glob("val_scores_epoch_*.csv")):
            match = re.search(r"epoch_(\d+)", val_path.name)
            if not match:
                continue
            epoch = int(match.group(1))
            ckpt_path = lam_dir / "checkpoints" / f"epoch_{epoch:04d}.pth"
            if not ckpt_path.exists():
                continue
            val = pd.read_csv(val_path)
            kadid_val = val[val["dataset"].eq(KADID)].copy()
            row = {
                "lambda_rank": lam,
                "epoch": epoch,
                "ckpt_path": str(ckpt_path),
                "stored_epoch": stored_epoch(ckpt_path, device),
            }
            s_h = 0.0
            for key, dtype in LOCKED_TYPES.items():
                sub = kadid_val[kadid_val["distortion_type"].eq(dtype)]
                sr, p, n, pr, pp = finite_corr(sub["E_H_ranked"], sub["severity_or_level"])
                row[f"{key}_SRCC"] = sr
                row[f"{key}_p"] = p
                row[f"{key}_N"] = n
                row[f"{key}_Pearson"] = pr
                s_h += sr
            row["S_H"] = s_h
            rows.append(row)

    table = pd.DataFrame(rows).sort_values(["lambda_rank", "epoch"]).reset_index(drop=True)
    table["best_within_lambda"] = "no"
    for lam, group in table.groupby("lambda_rank"):
        table.loc[group["S_H"].idxmax(), "best_within_lambda"] = "yes"
    table["selected_global"] = "no"
    table.loc[table["S_H"].idxmax(), "selected_global"] = "yes"
    winner = table[table["selected_global"].eq("yes")].iloc[0].to_dict()
    return table, winner


def choose_day11_ranked_ea() -> dict:
    sel = pd.read_csv(DAY_ROOT / "val_selection_table.csv")
    winners = sel[sel["selected"].astype(str).str.lower().eq("yes")].copy()
    if winners.empty:
        raise RuntimeError("No selected Day11 ranked-EA rows found.")
    chosen = winners.loc[winners["selection_score"].idxmax()].to_dict()
    chosen["stored_epoch"] = int(chosen["epoch"])
    return chosen


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    positive_table, eh_winner = build_positive_eh_selection(device)
    positive_table.to_csv(OUT_ROOT / "positive_EH_selection_table.csv", index=False)
    ea_winner = choose_day11_ranked_ea()

    manifest = day11.load_manifest()
    kadid_val = manifest[manifest["split"].eq("val") & manifest["dataset"].eq(KADID)].copy().reset_index(drop=True)
    kadid_holdout = manifest[manifest["split"].eq("holdout") & manifest["dataset"].eq(KADID)].copy().reset_index(drop=True)

    ea_lam = float(ea_winner["lambda_rank"])
    ea_epoch = int(ea_winner["epoch"])
    ea_ckpt = Path(str(ea_winner["ckpt_path"]))
    ea_val_path = DAY_ROOT / "day11_rankall" / f"lambda_{ea_lam}" / f"val_scores_epoch_{ea_epoch:04d}.csv"
    val_ea = pd.read_csv(ea_val_path)
    val_ea = val_ea[val_ea["dataset"].eq(KADID)].copy().reset_index(drop=True)
    holdout_ea = day11.score_ckpt_frame(kadid_holdout, ea_ckpt, device)

    eh_lam = float(eh_winner["lambda_rank"])
    eh_epoch = int(eh_winner["epoch"])
    eh_ckpt = Path(str(eh_winner["ckpt_path"]))
    eh_val_path = TASK12_ROOT / f"lambda_{eh_lam}" / f"val_scores_epoch_{eh_epoch:04d}.csv"
    val_eh = pd.read_csv(eh_val_path)
    val_eh = val_eh[val_eh["dataset"].eq(KADID)].copy().reset_index(drop=True)
    holdout_eh = task12.score_frame(kadid_holdout, eh_ckpt, device)

    val_ea_out = OUT_ROOT / f"val_ranked_EA_l{ea_lam}_ep{ea_epoch:04d}.csv"
    val_eh_out = OUT_ROOT / f"val_ranked_EH_l{eh_lam}_ep{eh_epoch:04d}.csv"
    hold_ea_out = OUT_ROOT / f"holdout_ranked_EA_l{ea_lam}_ep{ea_epoch:04d}.csv"
    hold_eh_out = OUT_ROOT / f"holdout_ranked_EH_l{eh_lam}_ep{eh_epoch:04d}.csv"
    val_ea.to_csv(val_ea_out, index=False)
    val_eh.to_csv(val_eh_out, index=False)
    holdout_ea.to_csv(hold_ea_out, index=False)
    holdout_eh.to_csv(hold_eh_out, index=False)

    ea_stats = z_stats(val_ea["E_A"])
    eh_stats = z_stats(val_eh["E_H_ranked"])
    norm = {
        "ranked_EA": {"lambda": ea_lam, "epoch": ea_epoch, "ckpt_path": str(ea_ckpt), **ea_stats},
        "ranked_EH": {"lambda": eh_lam, "epoch": eh_epoch, "ckpt_path": str(eh_ckpt), **eh_stats},
        "normalization_split": "KADID validation only",
        "std_definition": "population std, ddof=0",
    }
    (OUT_ROOT / "normalization_stats.json").write_text(json.dumps(norm, indent=2) + "\n")

    # Keep the Day-8/Day-11 context columns, but do not use stale ranked-EH columns.
    day11_hold = pd.read_csv(DAY_ROOT / "holdout_rankall_scores.csv")
    base_cols = [
        "image_id", "E_H", "S0_E_A_ep05", "z_H", "S0_z_A_ep05", "S0_Qz_AmH_ep05",
        "S0_Qz_HmA_ep05", "mos_or_dmos", "distortion_type", "severity_or_level",
    ]
    out = day11_hold[base_cols].copy()
    meta_cols = ["image_id", "dataset", "ref_id"]
    out = out.merge(kadid_holdout[meta_cols], on="image_id", how="left", validate="one_to_one")
    out = out.merge(
        holdout_ea[["image_id", "E_A"]].rename(columns={"E_A": "ranked_EA_raw"}),
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        holdout_eh[["image_id", "E_H_ranked"]].rename(columns={"E_H_ranked": "ranked_EH_raw"}),
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    out["z_ranked_EA"] = add_z(out["ranked_EA_raw"], ea_stats)
    out["z_ranked_EH"] = add_z(out["ranked_EH_raw"], eh_stats)
    out["Qz_uniform_rankedEA_minus_rankedEH"] = out["z_ranked_EA"] - out["z_ranked_EH"]
    out["Qz_uniform_rankedEH_minus_rankedEA"] = out["z_ranked_EH"] - out["z_ranked_EA"]
    ordered = [
        "dataset", "image_id", "ref_id", "distortion_type", "severity_or_level", "mos_or_dmos",
        "E_H", "S0_E_A_ep05", "S0_Qz_AmH_ep05", "ranked_EA_raw", "ranked_EH_raw",
        "z_ranked_EA", "z_ranked_EH", "Qz_uniform_rankedEA_minus_rankedEH",
        "Qz_uniform_rankedEH_minus_rankedEA",
    ]
    out = out[ordered]
    out.to_csv(OUT_ROOT / "corrected_ranked_dual_holdout.csv", index=False)

    metric_rows = []
    for name, col, target, dtype in [
        ("S0_Qz_AmH_ep05_vs_MOS", "S0_Qz_AmH_ep05", "mos_or_dmos", None),
        ("ranked_EA_vs_MOS", "ranked_EA_raw", "mos_or_dmos", None),
        ("selected_ranked_EH_vs_MOS", "ranked_EH_raw", "mos_or_dmos", None),
        ("Qz_uniform_EA_minus_EH_vs_MOS", "Qz_uniform_rankedEA_minus_rankedEH", "mos_or_dmos", None),
        ("Qz_uniform_EH_minus_EA_vs_MOS", "Qz_uniform_rankedEH_minus_rankedEA", "mos_or_dmos", None),
    ]:
        sub = out if dtype is None else out[out["distortion_type"].eq(dtype)]
        sr, sp, n, pr, pp = finite_corr(sub[col], sub[target])
        metric_rows.append({"scope": "KADID_holdout", "metric": name, "distortion_type": dtype or "all", "N": n, "Spearman": sr, "Spearman_p": sp, "Pearson": pr, "Pearson_p": pp})
    for key, dtype in LOCKED_TYPES.items():
        for label, col in [("ranked_EA_severity", "ranked_EA_raw"), ("ranked_EH_severity", "ranked_EH_raw")]:
            sub = out[out["distortion_type"].eq(dtype)]
            sr, sp, n, pr, pp = finite_corr(sub[col], sub["severity_or_level"])
            metric_rows.append({"scope": "KADID_holdout", "metric": label, "distortion_type": dtype, "N": n, "Spearman": sr, "Spearman_p": sp, "Pearson": pr, "Pearson_p": pp})
    corr = pd.DataFrame(metric_rows)
    corr.to_csv(OUT_ROOT / "corrected_baseline_correlations.csv", index=False)

    selected = {
        "ranked_EA": {"lambda": ea_lam, "epoch": ea_epoch, "stored_epoch": ea_winner["stored_epoch"], "ckpt_path": str(ea_ckpt), "selection_source": str(DAY_ROOT / "val_selection_table.csv")},
        "ranked_EH": {"lambda": eh_lam, "epoch": eh_epoch, "stored_epoch": eh_winner["stored_epoch"], "ckpt_path": str(eh_ckpt), "selection_rule": "max S_H = blur + lens + sharpen + pixelate validation SRCC"},
        "expected_ranked_EH_winner_confirmed": bool(eh_lam == 0.1 and eh_epoch == 20),
    }
    (OUT_ROOT / "selected_checkpoints.json").write_text(json.dumps(selected, indent=2) + "\n")

    q_row = corr[corr["metric"].eq("Qz_uniform_EA_minus_EH_vs_MOS")].iloc[0]
    report = [
        "Stage0 corrected baseline report",
        "================================",
        "",
        "Fix applied:",
        "The ranked_E_H holdout scores were freshly recomputed from the selected checkpoint.",
        "No stale Day-11b holdout ranked_E_H_lambda_0.1 column was reused.",
        "",
        f"Device used: {device}",
        "",
        "Selected ranked_E_H checkpoint by positive-E_H validation rule:",
        f"  lambda={eh_lam}, epoch={eh_epoch}, stored_epoch={eh_winner['stored_epoch']}",
        f"  path={eh_ckpt}",
        f"  S_H={eh_winner['S_H']:.6f}",
        f"  expected lambda=0.1 epoch=20 confirmed: {selected['expected_ranked_EH_winner_confirmed']}",
        "",
        "Selected ranked_E_A checkpoint from Day-11 selected validation table:",
        f"  lambda={ea_lam}, epoch={ea_epoch}, stored_epoch={ea_winner['stored_epoch']}",
        f"  path={ea_ckpt}",
        "",
        "Corrected KADID holdout result:",
        f"  Qz_uniform = z(E_A_rankall) - z(E_H_ranked)",
        f"  Spearman vs MOS = {q_row['Spearman']:.6f}, Pearson vs MOS = {q_row['Pearson']:.6f}, N={int(q_row['N'])}",
        "",
        "Outputs:",
        *[f"  {p.name}" for p in sorted(OUT_ROOT.iterdir()) if p.is_file()],
    ]
    (OUT_ROOT / "stage0_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
