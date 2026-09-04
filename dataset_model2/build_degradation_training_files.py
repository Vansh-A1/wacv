#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("/home/projectwork/student_package/dataset_model2")
SPLIT_DIR = ROOT / "ref_splits_seed42"
OUT_DIR = ROOT / "training_manifests"
COMBINED = SPLIT_DIR / "combined_kadid_tid_koniq_split_seed42.csv"
PATH_REMAPS = {
    "/data/projectwork/swati_mam/kadid10k": "/data/projectwork/swati_mam/new model data/kadid10k",
}


def remap_path(path: str) -> str:
    for old, new in PATH_REMAPS.items():
        if path.startswith(old):
            return new + path[len(old) :]
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(COMBINED, dtype=str, keep_default_na=False)
    df["distorted_path"] = df["distorted_path"].map(remap_path)
    df["ref_path"] = df["ref_path"].map(lambda p: remap_path(p) if p != "NA" else p)
    df = df[df["distorted_path"].map(lambda p: Path(p).is_file())].copy()

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    kadid_val = val[(val["dataset"] == "KADID-10k") & (val["mos_or_dmos"] != "NA")].copy()

    train_txt = OUT_DIR / "degradation_train_seed42.txt"
    train_csv = OUT_DIR / "degradation_train_seed42.csv"
    eval_txt = OUT_DIR / "degradation_val_eval_seed42.txt"
    kadid_monitor = OUT_DIR / "kadid_val_monitor_seed42.csv"
    summary_json = OUT_DIR / "degradation_training_files_summary.json"
    summary_txt = OUT_DIR / "degradation_training_files_summary.txt"

    train["distorted_path"].to_csv(train_txt, index=False, header=False)
    pd.DataFrame(
        {
            "path": train["distorted_path"],
            "pool": train["dataset"],
        }
    ).to_csv(train_csv, index=False)

    val["distorted_path"].to_csv(eval_txt, index=False, header=False)
    pd.DataFrame(
        {
            "path": kadid_val["distorted_path"],
            "ref_img": kadid_val["ref_path"],
            "mos": kadid_val["mos_or_dmos"],
        }
    ).to_csv(kadid_monitor, index=False)

    counts = {
        "train_total": int(len(train)),
        "val_eval_total": int(len(val)),
        "kadid_val_monitor_total": int(len(kadid_val)),
        "train_by_dataset": train["dataset"].value_counts().sort_index().astype(int).to_dict(),
        "val_by_dataset": val["dataset"].value_counts().sort_index().astype(int).to_dict(),
    }
    outputs = {
        "train_txt": str(train_txt),
        "train_csv": str(train_csv),
        "eval_txt": str(eval_txt),
        "kadid_monitor_csv": str(kadid_monitor),
        "summary_json": str(summary_json),
        "summary_txt": str(summary_txt),
    }
    summary = {
        "source_manifest": str(COMBINED),
        "output_dir": str(OUT_DIR),
        "counts": counts,
        "outputs": outputs,
        "notes": [
            "Training list contains only split=train distorted/authentic images.",
            "Eval list contains split=val images for reconstruction/latent diagnostics.",
            "KADID val monitor is used for kadid_srcc checkpoint selection because it has MOS.",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "Degradation WACV training files",
        f"Source manifest: {COMBINED}",
        f"Output directory: {OUT_DIR}",
        "",
        f"Train total: {counts['train_total']}",
        f"Val eval total: {counts['val_eval_total']}",
        f"KADID val monitor total: {counts['kadid_val_monitor_total']}",
        "",
        "Train by dataset:",
    ]
    lines.extend(f"- {k}: {v}" for k, v in counts["train_by_dataset"].items())
    lines.append("")
    lines.append("Val by dataset:")
    lines.extend(f"- {k}: {v}" for k, v in counts["val_by_dataset"].items())
    lines.append("")
    lines.append("Generated files:")
    lines.extend(f"- {v}" for v in outputs.values())
    summary_txt.write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
