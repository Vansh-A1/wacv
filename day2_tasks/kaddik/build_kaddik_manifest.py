#!/usr/bin/env python3
"""Build a KADID-10k manifest from the local dataset copy."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


DATASET_ROOT = Path("/data/projectwork/swati_mam/kadid10k")
IMAGES_DIR = DATASET_ROOT / "images"
DMOS_CSV = DATASET_ROOT / "dmos.csv"
OUT_DIR = Path("/home/projectwork/student_package/day2_tasks/kaddik")

DISTORTION_TYPES = {
    "01": "Gaussian blur",
    "02": "Lens blur",
    "03": "Motion blur",
    "04": "Color diffusion",
    "05": "Color shift",
    "06": "Color quantization",
    "07": "Color saturation 1",
    "08": "Color saturation 2",
    "09": "JPEG2000",
    "10": "JPEG",
    "11": "White noise",
    "12": "White noise in color component",
    "13": "Impulse noise",
    "14": "Multiplicative noise",
    "15": "Denoise",
    "16": "Brighten",
    "17": "Darken",
    "18": "Mean shift",
    "19": "Jitter",
    "20": "Non-eccentricity patch",
    "21": "Pixelate",
    "22": "Quantization",
    "23": "Color block",
    "24": "High sharpen",
    "25": "Contrast change",
}


def parse_distorted_name(filename: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"(I\d{2})_(\d{2})_(\d{2})\.png", filename)
    if not match:
        raise ValueError(f"Unexpected KADID distorted filename: {filename}")
    ref_id, distortion_code, severity = match.groups()
    return ref_id, distortion_code, severity


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DMOS_CSV.exists():
        raise FileNotFoundError(DMOS_CSV)
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(IMAGES_DIR)

    dmos = pd.read_csv(DMOS_CSV)
    required = {"dist_img", "ref_img", "dmos"}
    missing_cols = sorted(required - set(dmos.columns))
    if missing_cols:
        raise RuntimeError(f"Missing required columns in dmos.csv: {missing_cols}")

    rows = []
    missing_files = []
    parse_errors = []
    for rec in dmos.to_dict("records"):
        try:
            ref_id, distortion_code, severity = parse_distorted_name(rec["dist_img"])
        except ValueError as exc:
            parse_errors.append({"dist_img": rec["dist_img"], "error": str(exc)})
            continue

        distorted_path = IMAGES_DIR / rec["dist_img"]
        ref_path = IMAGES_DIR / rec["ref_img"]
        if not distorted_path.exists():
            missing_files.append(str(distorted_path))
        if not ref_path.exists():
            missing_files.append(str(ref_path))

        rows.append(
            {
                "dataset": "KADID-10k",
                "ref_id": ref_id,
                "distorted_path": str(distorted_path),
                "ref_path": str(ref_path),
                "distortion_type": DISTORTION_TYPES.get(distortion_code, f"unknown_{distortion_code}"),
                "severity_or_level": int(severity),
                "mos_or_dmos": float(rec["dmos"]),
                "distortion_code": distortion_code,
                "distorted_id": Path(rec["dist_img"]).stem,
                "distorted_filename": rec["dist_img"],
                "ref_filename": rec["ref_img"],
                "dmos_variance": float(rec["var"]) if "var" in rec and pd.notna(rec["var"]) else None,
            }
        )

    full = pd.DataFrame(rows)
    supervisor_columns = [
        "dataset",
        "ref_id",
        "distorted_path",
        "ref_path",
        "distortion_type",
        "severity_or_level",
        "mos_or_dmos",
    ]
    manifest = full[supervisor_columns].copy()

    manifest_path = OUT_DIR / "kadid10k_manifest.csv"
    full_path = OUT_DIR / "kadid10k_manifest_extended.csv"
    distortion_map_path = OUT_DIR / "kadid10k_distortion_code_map.csv"
    summary_path = OUT_DIR / "kadid10k_manifest_summary.json"
    readme_path = OUT_DIR / "README_kaddik.txt"

    manifest.to_csv(manifest_path, index=False)
    full.to_csv(full_path, index=False)
    pd.DataFrame(
        [{"distortion_code": k, "distortion_type": v} for k, v in DISTORTION_TYPES.items()]
    ).to_csv(distortion_map_path, index=False)

    reference_images = sorted(p for p in IMAGES_DIR.glob("I??.png") if not p.name.startswith("._"))
    distorted_images = sorted(p for p in IMAGES_DIR.glob("I??_??_??.png") if not p.name.startswith("._"))
    summary = {
        "task": "Steps 2 and 3: KADID-10k local dataset check and manifest",
        "dataset_root": str(DATASET_ROOT),
        "images_dir": str(IMAGES_DIR),
        "dmos_csv": str(DMOS_CSV),
        "output_dir": str(OUT_DIR),
        "manifest_csv": str(manifest_path),
        "manifest_extended_csv": str(full_path),
        "distortion_code_map_csv": str(distortion_map_path),
        "num_manifest_rows": int(len(manifest)),
        "num_reference_images_found": int(len(reference_images)),
        "num_distorted_images_found": int(len(distorted_images)),
        "num_distortion_types": int(full["distortion_code"].nunique()) if len(full) else 0,
        "severity_levels": sorted(int(x) for x in full["severity_or_level"].unique()) if len(full) else [],
        "missing_files_count": int(len(set(missing_files))),
        "missing_files_sample": sorted(set(missing_files))[:20],
        "parse_errors_count": int(len(parse_errors)),
        "parse_errors_sample": parse_errors[:20],
        "dmos_min": float(full["mos_or_dmos"].min()) if len(full) else None,
        "dmos_max": float(full["mos_or_dmos"].max()) if len(full) else None,
        "notes": [
            "Main manifest uses the exact requested columns.",
            "mos_or_dmos contains the KADID-10k dmos score from dmos.csv.",
            "AppleDouble files beginning with ._ are ignored.",
            "Distortion type names follow the official KADID-10k code order.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    readme_path.write_text(
        "\n".join(
            [
                "KADID-10k manifest outputs",
                "",
                f"Dataset root: {DATASET_ROOT}",
                f"Output directory: {OUT_DIR}",
                "",
                "Created files:",
                f"- {manifest_path.name}: exact requested columns",
                f"- {full_path.name}: requested columns plus filename/code/variance audit columns",
                f"- {distortion_map_path.name}: distortion code to distortion type",
                f"- {summary_path.name}: count and validation summary",
                "",
                "Requested manifest columns:",
                "dataset, ref_id, distorted_path, ref_path, distortion_type, severity_or_level, mos_or_dmos",
                "",
                "Filename parsing:",
                "I01_03_05.png -> ref_id=I01, distortion_code=03, severity_or_level=5.",
                "mos_or_dmos is DMOS from the local dmos.csv.",
                "",
                "No image files were modified.",
            ]
        )
        + "\n"
    )

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {full_path}")
    print(f"Rows: {len(manifest)}")
    print(f"Reference images found: {len(reference_images)}")
    print(f"Distorted images found: {len(distorted_images)}")
    print(f"Missing files: {len(set(missing_files))}")


if __name__ == "__main__":
    main()
