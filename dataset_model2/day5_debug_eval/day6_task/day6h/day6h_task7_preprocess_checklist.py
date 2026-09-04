#!/usr/bin/env python3
"""Day 6 Task 7: preprocessing checklist for old Day-1/2 vs Day-5 E_H."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path("/home/projectwork/student_package")
DAY5_DIR = PACKAGE_ROOT / "dataset_model2" / "day5_debug_eval"
OUT_DIR = DAY5_DIR / "day6h"

OUT_CSV = OUT_DIR / "task7_preprocess_checklist.csv"
OUT_JSON = OUT_DIR / "task7_preprocess_checklist.json"
OUT_TXT = OUT_DIR / "task7_preprocess_checklist_report.txt"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "Preprocess item": "Resize rule",
            "Old Day-1/2": "yes: resize short side to >=256 before scoring",
            "Day-5 E_H": "yes: resize short side to >=256 before scoring",
            "match": "yes",
            "evidence": "task wacv/check_* scripts and day5 diagnostics use _resize_short_side(..., 256).",
        },
        {
            "Preprocess item": "Crop",
            "Old Day-1/2": "center 256 crop for scoring",
            "Day-5 E_H": "center 256 crop for scoring",
            "match": "yes",
            "evidence": "Old check scripts call center_crop after resize; diagnostics.json says Day-5 E_H uses center 256x256 crop.",
        },
        {
            "Preprocess item": "Color",
            "Old Day-1/2": "RGB",
            "Day-5 E_H": "RGB",
            "match": "yes",
            "evidence": "Image.open(...).convert('RGB') is used in scoring scripts.",
        },
        {
            "Preprocess item": "Value range / normalization",
            "Old Day-1/2": "[0,1], no ImageNet mean/std for WACV input",
            "Day-5 E_H": "[0,1], no ImageNet mean/std for WACV input",
            "match": "yes",
            "evidence": "Scoring converts np.asarray(image)/255.0 then ToTensor; diagnostics.json says no mean/std normalization.",
        },
        {
            "Preprocess item": "Flip at test",
            "Old Day-1/2": "no",
            "Day-5 E_H": "no",
            "match": "yes",
            "evidence": "Evaluation/scoring uses deterministic center crop only; no hflip in test/inference path.",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    summary = {
        "task": "Task 7: preprocessing checklist",
        "conclusion": "Old Day-1/2 WACV scoring and Day-5 E_H scoring use matching test-time preprocessing.",
        "source_files_checked": [
            str(PACKAGE_ROOT / "task wacv" / "check_a_clean_vs_degraded.py"),
            str(PACKAGE_ROOT / "task wacv" / "check_b_severity_trend.py"),
            str(PACKAGE_ROOT / "task wacv" / "check_c_ballpark_pristine_test.py"),
            str(PACKAGE_ROOT / "infer.py"),
            str(DAY5_DIR / "diagnostics.json"),
        ],
        "outputs": {
            "checklist_csv": str(OUT_CSV),
            "summary_json": str(OUT_JSON),
            "report_txt": str(OUT_TXT),
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "Day 6 Task 7: preprocessing checklist",
        "",
        "Conclusion: old Day-1/2 WACV scoring and Day-5 E_H scoring match at test time.",
        "",
        df.to_string(index=False),
        "",
        "Important distinction:",
        "- Training-time degradation model uses random crop and hflip.",
        "- This checklist is about scoring/evaluation preprocessing, where both old WACV and Day-5 E_H use deterministic center crop.",
        "",
        "Outputs:",
        f"- {OUT_CSV}",
        f"- {OUT_JSON}",
        f"- {OUT_TXT}",
    ]
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"Saved Task 7 outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
