#!/usr/bin/env python3
"""Reference-level train/val/holdout splits for KADID-10k and TID2013."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd


SPLIT_SEED = 42
OUT_DIR = Path("/home/projectwork/student_package/day2_tasks/ref_splits_seed42")
KADID_MANIFEST = Path("/home/projectwork/student_package/day2_tasks/kaddik/kadid10k_manifest.csv")
TID_ROOT = Path("/data/projectwork/swati_mam/new model data/tid2013/tid2013")


def split_ref_ids(ref_ids: list[str]) -> dict[str, str]:
    ordered = sorted(ref_ids)
    shuffled = ordered[:]
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    split_map = {}
    for ref_id in shuffled[:n_train]:
        split_map[ref_id] = "train"
    for ref_id in shuffled[n_train : n_train + n_val]:
        split_map[ref_id] = "val"
    for ref_id in shuffled[n_train + n_val :]:
        split_map[ref_id] = "holdout"
    return split_map


def write_ref_split_csv(dataset: str, split_map: dict[str, str], path: Path) -> None:
    rows = [
        {"dataset": dataset, "ref_id": ref_id, "split": split_map[ref_id]}
        for ref_id in sorted(split_map)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def build_kadid() -> tuple[pd.DataFrame, dict[str, str]]:
    manifest = pd.read_csv(KADID_MANIFEST)
    split_map = split_ref_ids(sorted(manifest["ref_id"].unique()))
    out = manifest.copy()
    out["split"] = out["ref_id"].map(split_map)
    return out, split_map


def tid_ref_map() -> dict[str, Path]:
    ref_dir = TID_ROOT / "reference_images"
    refs = {}
    for path in ref_dir.iterdir():
        if not path.is_file() or path.name.lower() == "desktop.ini":
            continue
        match = re.fullmatch(r"i(\d{2})\.bmp", path.name, flags=re.IGNORECASE)
        if match:
            refs[f"I{match.group(1)}"] = path
    return refs


def build_tid2013() -> tuple[pd.DataFrame, dict[str, str]]:
    dist_dir = TID_ROOT / "distorted_images"
    mos_path = TID_ROOT / "mos.csv"
    mos = pd.read_csv(mos_path)
    mos_by_name = {str(row["image_id"]).lower(): row for row in mos.to_dict("records")}
    ref_paths = tid_ref_map()
    rows = []
    pattern = re.compile(r"^[iI](\d{2})_(\d{2})_(\d)\.bmp$")
    for path in sorted(dist_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() != ".bmp":
            continue
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        ref_num, distortion_code, level = match.groups()
        ref_id = f"I{ref_num}"
        if ref_id not in ref_paths:
            raise RuntimeError(f"Missing reference image for {ref_id}")
        mos_key = path.name.lower()
        if mos_key not in mos_by_name:
            raise RuntimeError(f"Missing MOS row for {path.name}")
        mos_row = mos_by_name[mos_key]
        rows.append(
            {
                "dataset": "TID2013",
                "ref_id": ref_id,
                "distorted_path": str(path),
                "ref_path": str(ref_paths[ref_id]),
                "distortion_type": f"type_{distortion_code}",
                "severity_or_level": int(level),
                "mos_or_dmos": float(mos_row["mean"]),
                "mos_std": float(mos_row["std"]),
                "distortion_code": distortion_code,
                "distorted_filename": path.name,
            }
        )
    manifest = pd.DataFrame(rows)
    split_map = split_ref_ids(sorted(manifest["ref_id"].unique()))
    manifest["split"] = manifest["ref_id"].map(split_map)
    return manifest, split_map


def verify_reference_exclusivity(manifest: pd.DataFrame) -> dict[str, object]:
    grouped = manifest.groupby("ref_id")["split"].nunique()
    bad = grouped[grouped != 1]
    counts_refs = manifest[["ref_id", "split"]].drop_duplicates()["split"].value_counts().to_dict()
    counts_rows = manifest["split"].value_counts().to_dict()
    return {
        "num_refs": int(grouped.shape[0]),
        "row_counts_by_split": {k: int(v) for k, v in counts_rows.items()},
        "ref_counts_by_split": {k: int(v) for k, v in counts_refs.items()},
        "num_refs_in_multiple_splits": int(len(bad)),
        "refs_in_multiple_splits": bad.index.tolist(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kadid, kadid_split = build_kadid()
    tid, tid_split = build_tid2013()

    kadid_manifest_path = OUT_DIR / "kadid10k_manifest_ref_split_seed42.csv"
    tid_manifest_path = OUT_DIR / "tid2013_manifest_ref_split_seed42.csv"
    kadid_refs_path = OUT_DIR / "kadid10k_ref_splits_seed42.csv"
    tid_refs_path = OUT_DIR / "tid2013_ref_splits_seed42.csv"
    combined_path = OUT_DIR / "combined_kadid_tid_ref_split_seed42.csv"

    kadid.to_csv(kadid_manifest_path, index=False)
    tid.to_csv(tid_manifest_path, index=False)
    write_ref_split_csv("KADID-10k", kadid_split, kadid_refs_path)
    write_ref_split_csv("TID2013", tid_split, tid_refs_path)

    combined_cols = [
        "dataset",
        "ref_id",
        "distorted_path",
        "ref_path",
        "distortion_type",
        "severity_or_level",
        "mos_or_dmos",
        "split",
    ]
    pd.concat([kadid[combined_cols], tid[combined_cols]], ignore_index=True).to_csv(combined_path, index=False)

    summary = {
        "task": "Reference-level train/val/holdout splits for KADID-10k and TID2013",
        "split_seed": SPLIT_SEED,
        "procedure": [
            "List unique ref_id values.",
            "Sort ref_ids lexicographically.",
            "Shuffle with random.Random(split_seed=42).",
            "Assign first int(70%) refs to train, next int(15%) refs to val, remainder to holdout.",
            "Every distorted image inherits its ref_id split.",
        ],
        "outputs": {
            "kadid_manifest_with_split": str(kadid_manifest_path),
            "tid2013_manifest_with_split": str(tid_manifest_path),
            "kadid_ref_splits": str(kadid_refs_path),
            "tid2013_ref_splits": str(tid_refs_path),
            "combined_manifest_with_split": str(combined_path),
            "summary_json": str(OUT_DIR / "reference_split_summary.json"),
            "summary_txt": str(OUT_DIR / "reference_split_summary.txt"),
        },
        "kadid": verify_reference_exclusivity(kadid),
        "tid2013": verify_reference_exclusivity(tid),
    }
    (OUT_DIR / "reference_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "Reference-level splits: KADID-10k and TID2013",
        f"Split seed: {SPLIT_SEED}",
        "Method: sort ref_ids lexicographically, shuffle with random.Random(42), train=int(70%), val=int(15%), holdout=remainder.",
        "",
        f"KADID refs by split: {summary['kadid']['ref_counts_by_split']}",
        f"KADID rows by split: {summary['kadid']['row_counts_by_split']}",
        f"KADID refs in multiple splits: {summary['kadid']['num_refs_in_multiple_splits']}",
        "",
        f"TID2013 refs by split: {summary['tid2013']['ref_counts_by_split']}",
        f"TID2013 rows by split: {summary['tid2013']['row_counts_by_split']}",
        f"TID2013 refs in multiple splits: {summary['tid2013']['num_refs_in_multiple_splits']}",
        "",
        "Outputs:",
    ]
    lines.extend(f"- {v}" for v in summary["outputs"].values())
    (OUT_DIR / "reference_split_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
