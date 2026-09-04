#!/usr/bin/env python3
"""Find images where LR_content and D_WACV disagree strongly."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd


INPUTS = {
    "SwinSR": Path("/home/projectwork/student_package/day2_tasks/task_e_metric_correspondence_negated_psnr_ssim/task_e_swinsr_metrics.csv"),
    "HAT": Path("/home/projectwork/student_package/day2_tasks/task_e_metric_correspondence_negated_psnr_ssim/task_e_hat_metrics.csv"),
}
OUT_DIR = Path("/home/projectwork/student_package/day2_tasks/lrcontent_wacv_contradiction_images")
TOP_K = 5
METRIC_COLUMNS = ["D_WACV", "LR_content", "NIQE", "PSNR", "SSIM", "LPIPS", "DISTS"]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=1)
    if pd.isna(std) or std == 0:
        return series * 0.0
    return (series - series.mean()) / std


def safe_copy(src: str, dst: Path) -> str:
    src_path = Path(src)
    if not src_path.exists():
        return "missing"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return str(dst)


def select_cases(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = df.copy()
    work["D_WACV_z"] = zscore(work["D_WACV"])
    work["LR_content_z"] = zscore(work["LR_content"])
    work["low_lrcontent_high_wacv_score"] = work["D_WACV_z"] - work["LR_content_z"]
    work["low_wacv_high_lrcontent_score"] = work["LR_content_z"] - work["D_WACV_z"]
    return {
        "low_lrcontent_high_wacv": work.sort_values("low_lrcontent_high_wacv_score", ascending=False).head(TOP_K),
        "low_wacv_high_lrcontent": work.sort_values("low_wacv_high_lrcontent_score", ascending=False).head(TOP_K),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []

    for method, csv_path in INPUTS.items():
        df = pd.read_csv(csv_path)
        cases = select_cases(df)
        method_dir = OUT_DIR / method
        method_dir.mkdir(parents=True, exist_ok=True)

        for case_name, selected in cases.items():
            case_dir = method_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            selected.to_csv(case_dir / "selected_rows.csv", index=False)

            for rank, row in enumerate(selected.to_dict("records"), start=1):
                image_id = row["image_id"]
                img_dir = case_dir / f"{rank:02d}_{image_id}"
                img_dir.mkdir(parents=True, exist_ok=True)

                copied = {
                    "sr_copy": safe_copy(row["sr_path"], img_dir / f"{image_id}_{method}_SR.png"),
                    "hr_copy": safe_copy(row["hr_path"], img_dir / f"{image_id}_HR.png"),
                    "lr_copy": safe_copy(row["lr_path"], img_dir / f"{image_id}_LR.png"),
                }
                metadata = {
                    "method": method,
                    "case": case_name,
                    "rank": rank,
                    "image_id": image_id,
                    "selection_rule": (
                        "low_lrcontent_high_wacv ranks by z(D_WACV) - z(LR_content); "
                        "low_wacv_high_lrcontent ranks by z(LR_content) - z(D_WACV), computed within each method CSV."
                    ),
                    "source_paths": {
                        "sr_path": row["sr_path"],
                        "hr_path": row["hr_path"],
                        "lr_path": row["lr_path"],
                    },
                    "copied_files": copied,
                    "metrics": {col: float(row[col]) for col in METRIC_COLUMNS if col in row and pd.notna(row[col])},
                    "normalized_values": {
                        "D_WACV_z": float(row["D_WACV_z"]),
                        "LR_content_z": float(row["LR_content_z"]),
                    },
                    "contradiction_scores": {
                        "low_lrcontent_high_wacv_score": float(row["low_lrcontent_high_wacv_score"]),
                        "low_wacv_high_lrcontent_score": float(row["low_wacv_high_lrcontent_score"]),
                    },
                }
                (img_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                all_rows.append(
                    {
                        "method": method,
                        "case": case_name,
                        "rank": rank,
                        "image_id": image_id,
                        "image_folder": str(img_dir),
                        "sr_copy": copied["sr_copy"],
                        "hr_copy": copied["hr_copy"],
                        "lr_copy": copied["lr_copy"],
                        "D_WACV": row["D_WACV"],
                        "LR_content": row["LR_content"],
                        "D_WACV_z": row["D_WACV_z"],
                        "LR_content_z": row["LR_content_z"],
                        "low_lrcontent_high_wacv_score": row["low_lrcontent_high_wacv_score"],
                        "low_wacv_high_lrcontent_score": row["low_wacv_high_lrcontent_score"],
                    }
                )

    summary_csv = OUT_DIR / "contradiction_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    (OUT_DIR / "README.txt").write_text(
        "\n".join(
            [
                "LR_content vs D_WACV contradiction image selection",
                "",
                "For each method separately, D_WACV and LR_content were z-normalized within that method CSV.",
                "Case 1: low_lrcontent_high_wacv = high z(D_WACV) - z(LR_content).",
                "Case 2: low_wacv_high_lrcontent = high z(LR_content) - z(D_WACV).",
                f"Top K per method/case: {TOP_K}.",
                "",
                "Each selected image folder contains copied SR, HR, LR images and metadata.json.",
                "Original images were not modified.",
                f"Summary CSV: {summary_csv}",
            ]
        )
        + "\n"
    )
    print(f"Wrote contradiction images to {OUT_DIR}")
    print(f"Rows selected: {len(all_rows)}")


if __name__ == "__main__":
    main()
