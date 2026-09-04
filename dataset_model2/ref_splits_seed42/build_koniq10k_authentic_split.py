#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

KONIQ_DIR = Path("/data/projectwork/swati_mam/new model data/koniq10k_1024x768/1024x768")
OUT_DIR = Path("/home/projectwork/student_package/dataset_model2/ref_splits_seed42")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

KONIQ_SPLITS_CSV = OUT_DIR / "koniq10k_authentic_image_splits_seed42.csv"
KONIQ_MANIFEST_CSV = OUT_DIR / "koniq10k_authentic_manifest_split_seed42.csv"
COMBINED_IN_CSV = OUT_DIR / "combined_kadid_tid_ref_split_seed42.csv"
COMBINED_OUT_CSV = OUT_DIR / "combined_kadid_tid_koniq_split_seed42.csv"
SUMMARY_JSON = OUT_DIR / "koniq10k_authentic_split_summary.json"
SUMMARY_TXT = OUT_DIR / "koniq10k_authentic_split_summary.txt"


def assign_splits(paths: list[Path]) -> pd.DataFrame:
    items = sorted(((p.stem, p.name, str(p)) for p in paths), key=lambda row: row[0])
    shuffled = items[:]
    random.Random(SEED).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(TRAIN_FRAC * n_total)
    n_val = int(VAL_FRAC * n_total)

    split_by_id: dict[str, str] = {}
    for idx, (image_id, _filename, _path) in enumerate(shuffled):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "holdout"
        split_by_id[image_id] = split

    rows = []
    for image_id, filename, path in items:
        rows.append(
            {
                "dataset": "KONIQ-10k",
                "image_id": image_id,
                "filename": filename,
                "path": path,
                "split": split_by_id[image_id],
            }
        )
    return pd.DataFrame(rows)


def make_manifest(split_df: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.DataFrame(
        {
            "dataset": split_df["dataset"],
            "image_id": split_df["image_id"],
            "ref_id": "NA",
            "distorted_path": split_df["path"],
            "ref_path": "NA",
            "distortion_type": "authentic",
            "distortion_code": "NA",
            "severity_or_level": "NA",
            "mos_or_dmos": "NA",
            "split": split_df["split"],
        }
    )
    return manifest


def make_combined_manifest(koniq_manifest: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(COMBINED_IN_CSV)
    if "image_id" not in existing.columns:
        existing["image_id"] = existing["distorted_path"].map(lambda p: Path(str(p)).stem)
    if "distortion_code" not in existing.columns:
        existing["distortion_code"] = "NA"

    combined_cols = [
        "dataset",
        "image_id",
        "ref_id",
        "distorted_path",
        "ref_path",
        "distortion_type",
        "distortion_code",
        "severity_or_level",
        "mos_or_dmos",
        "split",
    ]
    for col in combined_cols:
        if col not in existing.columns:
            existing[col] = "NA"
        if col not in koniq_manifest.columns:
            koniq_manifest[col] = "NA"

    return pd.concat(
        [existing[combined_cols], koniq_manifest[combined_cols]],
        ignore_index=True,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = [p for p in KONIQ_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not image_paths:
        raise RuntimeError(f"No image files found in {KONIQ_DIR}")

    split_df = assign_splits(image_paths)
    manifest = make_manifest(split_df)
    combined = make_combined_manifest(manifest.copy())

    split_df.to_csv(KONIQ_SPLITS_CSV, index=False)
    manifest.to_csv(KONIQ_MANIFEST_CSV, index=False)
    combined.to_csv(COMBINED_OUT_CSV, index=False)

    split_counts = split_df["split"].value_counts().reindex(["train", "val", "holdout"]).fillna(0).astype(int)
    combined_counts = (
        combined.groupby(["dataset", "split"], dropna=False)
        .size()
        .reset_index(name="N")
        .sort_values(["dataset", "split"])
    )
    summary = {
        "dataset": "KONIQ-10k",
        "input_dir": str(KONIQ_DIR),
        "output_dir": str(OUT_DIR),
        "split_seed": SEED,
        "subset_used": False,
        "num_images": int(len(split_df)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "split_rule": "Sort image_ids lexicographically, shuffle with random.Random(42), assign first 70% train, next 15% val, last 15% holdout.",
        "manifest_notes": {
            "distortion_type": "authentic",
            "distortion_code": "NA",
            "severity_or_level": "NA",
            "ref_id": "NA",
            "ref_path": "NA",
            "mos_or_dmos": "NA",
        },
        "outputs": {
            "image_splits": str(KONIQ_SPLITS_CSV),
            "manifest": str(KONIQ_MANIFEST_CSV),
            "combined_manifest": str(COMBINED_OUT_CSV),
            "summary_json": str(SUMMARY_JSON),
            "summary_txt": str(SUMMARY_TXT),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "KONIQ-10k authentic image split, seed 42",
        f"Input directory: {KONIQ_DIR}",
        f"Output directory: {OUT_DIR}",
        "",
        "No paired pristine references are used for KONIQ-10k.",
        "Manifest fields are set as: distortion_type=authentic, distortion_code=NA, severity_or_level=NA, ref_id=NA, ref_path=NA, mos_or_dmos=NA.",
        "",
        f"Total images: {len(split_df)}",
        f"Train: {split_counts['train']}",
        f"Val: {split_counts['val']}",
        f"Holdout: {split_counts['holdout']}",
        "",
        "Split rule: list image_ids, sort lexicographically, shuffle with random.Random(split_seed=42), then assign 70% / 15% / 15%.",
        "",
        "Combined dataset row counts:",
        combined_counts.to_string(index=False),
        "",
        "Generated files:",
        f"- {KONIQ_SPLITS_CSV}",
        f"- {KONIQ_MANIFEST_CSV}",
        f"- {COMBINED_OUT_CSV}",
        f"- {SUMMARY_JSON}",
        f"- {SUMMARY_TXT}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
