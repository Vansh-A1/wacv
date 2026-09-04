#!/usr/bin/env python3
"""Reverse SR diagnostic: compare original LR with SR downsampled back to LR."""

from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_OUT = Path("/home/projectwork/student_package/day2_tasks/reverse_diagnostic_sr_down_vs_lr")
CROP_SIZE = 512
SCALE = 4
SEED = 42


@dataclass(frozen=True)
class ImageCase:
    folder: str
    image_id: str
    lr_path: Path
    sr_paths: dict[str, Path]


CASES = [
    ImageCase(
        folder="image_1",
        image_id="sony_160",
        lr_path=Path(
            "/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/"
            "DRealSR-20231109T095957Z-004/DRealSR/Test_x4/LR/dreal4_84.png"
        ),
        sr_paths={
            "swinsr": Path("/data/projectwork/sr_output/swinIr_Inference/SwinIR_results/Dreal/DRealSR_all_x4_sony_160_x1.png"),
            "hat": Path("/data/projectwork/sr_output/HAT_inference/DReal_x4/visualization/DReal_x4/sony_160_x1_DReal_x4.png"),
        },
    ),
    ImageCase(
        folder="image_2",
        image_id="Canon_10",
        lr_path=Path(
            "/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/"
            "DRealSR-20231109T095957Z-004/DRealSR/Test_x4/LR/dreal4_1.png"
        ),
        sr_paths={
            "swinsr": Path("/data/projectwork/sr_output/swinIr_Inference/SwinIR_results/Dreal/DRealSR_all_x4_Canon_10_x1.png"),
            "hat": Path("/data/projectwork/sr_output/HAT_inference/DReal_x4/visualization/DReal_x4/Canon_10_x1_DReal_x4.png"),
        },
    ),
]


SUMMARY_FIELDS = [
    "image_folder",
    "image_id",
    "model",
    "lr_path",
    "sr_path",
    "lr_width",
    "lr_height",
    "sr_width",
    "sr_height",
    "down_width",
    "down_height",
    "crop_x",
    "crop_y",
    "crop_size",
    "mean_absolute_difference",
    "median_absolute_difference",
    "std_absolute_difference",
    "max_absolute_difference",
    "percentage_pixels_absdiff_gt_1",
    "percentage_pixels_absdiff_gt_2",
    "percentage_pixels_absdiff_gt_5",
    "percentage_pixels_absdiff_gt_10",
]


def make_fresh_output_dir(base: Path) -> Path:
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = Path(f"{base}_run{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def snapshot(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def unchanged(before: dict[str, int | str]) -> bool:
    now = snapshot(Path(str(before["path"])))
    return now["size"] == before["size"] and now["mtime_ns"] == before["mtime_ns"]


def downsample_area(sr: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    arr = cv2.cvtColor(np.asarray(sr), cv2.COLOR_RGB2BGR)
    down = cv2.resize(arr, target_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(down, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def crop_coord(width: int, height: int, seed: int) -> tuple[int, int]:
    if width < CROP_SIZE or height < CROP_SIZE:
        raise ValueError(f"Image is too small for a {CROP_SIZE}x{CROP_SIZE} crop: {width}x{height}")
    rng = random.Random(seed)
    return rng.randint(0, width - CROP_SIZE), rng.randint(0, height - CROP_SIZE)


def signed_vis(diff: np.ndarray) -> Image.Image:
    max_abs = max(1, int(np.abs(diff).max()))
    vis = ((diff.astype(np.float32) + max_abs) * (255.0 / (2 * max_abs))).clip(0, 255).astype(np.uint8)
    return Image.fromarray(vis, mode="RGB")


def comparison(lr_crop: Image.Image, down_crop: Image.Image, abs_img: Image.Image, signed_img: Image.Image, out_path: Path) -> None:
    label_h = 34
    gap = 8
    panels = [lr_crop, down_crop, abs_img, signed_img]
    labels = ["Original LR crop", "SR downsampled crop", "|SR_down - LR|", "SR_down - LR"]
    w, h = lr_crop.size
    canvas = Image.new("RGB", (w * 4 + gap * 3, h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x0 = i * (w + gap)
        draw.text((x0 + 8, 10), label, fill=(0, 0, 0), font=font)
        canvas.paste(panel, (x0, label_h))
    canvas.save(out_path)


def stats(absdiff: np.ndarray) -> dict[str, float | int]:
    per_pixel = absdiff.max(axis=2)
    return {
        "mean_absolute_difference": float(absdiff.mean()),
        "median_absolute_difference": float(np.median(absdiff)),
        "std_absolute_difference": float(absdiff.std()),
        "max_absolute_difference": int(absdiff.max()),
        "percentage_pixels_absdiff_gt_1": float(100.0 * np.mean(per_pixel > 1)),
        "percentage_pixels_absdiff_gt_2": float(100.0 * np.mean(per_pixel > 2)),
        "percentage_pixels_absdiff_gt_5": float(100.0 * np.mean(per_pixel > 5)),
        "percentage_pixels_absdiff_gt_10": float(100.0 * np.mean(per_pixel > 10)),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = make_fresh_output_dir(BASE_OUT)
    out_dir.mkdir(parents=True)
    summary_rows: list[dict[str, object]] = []
    input_states: dict[str, dict[str, int | str]] = {}

    for case_index, case in enumerate(CASES, start=1):
        if not case.lr_path.exists():
            raise FileNotFoundError(case.lr_path)
        lr = pil_rgb(case.lr_path)
        x, y = crop_coord(lr.width, lr.height, SEED + case_index - 1)
        image_dir = out_dir / case.folder
        image_dir.mkdir()
        shutil.copy2(case.lr_path, image_dir / "lr_original.png")
        lr_crop = lr.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))
        lr_crop.save(image_dir / "lr_crop_512.png")
        input_states[f"{case.folder}_lr"] = snapshot(case.lr_path)

        image_metadata = {
            "image_id": case.image_id,
            "seed": SEED + case_index - 1,
            "crop_x": x,
            "crop_y": y,
            "crop_size": CROP_SIZE,
            "scale": SCALE,
            "lr_width": lr.width,
            "lr_height": lr.height,
            "models": {},
        }

        print(f"{case.folder} / {case.image_id}")
        print(f"  LR path: {case.lr_path}")
        print(f"  LR dimensions: {lr.width}x{lr.height}")
        print(f"  crop coordinates: x={x}, y={y}")

        for model, sr_path in case.sr_paths.items():
            if not sr_path.exists():
                raise FileNotFoundError(sr_path)
            input_states[f"{case.folder}_{model}_sr"] = snapshot(sr_path)
            sr = pil_rgb(sr_path)
            expected_sr = (lr.width * SCALE, lr.height * SCALE)
            if sr.size != expected_sr:
                raise ValueError(f"{case.folder} {model}: expected SR {expected_sr}, got {sr.size}")
            sr_down = downsample_area(sr, lr.size)
            if sr_down.size != lr.size:
                raise ValueError(f"{case.folder} {model}: downsampled SR {sr_down.size}, LR {lr.size}")

            model_dir = image_dir / model
            model_dir.mkdir()
            shutil.copy2(sr_path, model_dir / "sr_original.png")
            sr_down.save(model_dir / "sr_downsampled_x4.png")
            down_crop = sr_down.crop((x, y, x + CROP_SIZE, y + CROP_SIZE))
            down_crop.save(model_dir / "sr_downsampled_crop_512.png")
            lr_crop.save(model_dir / "lr_crop_512.png")

            lr_arr = np.asarray(lr_crop, dtype=np.int16)
            down_arr = np.asarray(down_crop, dtype=np.int16)
            diff = down_arr - lr_arr
            absdiff = np.abs(diff).astype(np.uint8)
            abs_img = Image.fromarray(absdiff, mode="RGB")
            signed_img = signed_vis(diff)

            np.save(model_dir / "difference_signed_raw.npy", diff)
            np.save(model_dir / "difference_abs_raw.npy", absdiff)
            abs_img.save(model_dir / "difference_abs.png")
            signed_img.save(model_dir / "difference_signed.png")
            comparison(lr_crop, down_crop, abs_img, signed_img, model_dir / "comparison.png")

            row = {
                "image_folder": case.folder,
                "image_id": case.image_id,
                "model": model,
                "lr_path": str(case.lr_path),
                "sr_path": str(sr_path),
                "lr_width": lr.width,
                "lr_height": lr.height,
                "sr_width": sr.width,
                "sr_height": sr.height,
                "down_width": sr_down.width,
                "down_height": sr_down.height,
                "crop_x": x,
                "crop_y": y,
                "crop_size": CROP_SIZE,
                **stats(absdiff),
            }
            summary_rows.append(row)
            model_meta = {
                "sr_path": str(sr_path),
                "sr_width": sr.width,
                "sr_height": sr.height,
                "downsampled_width": sr_down.width,
                "downsampled_height": sr_down.height,
                "downsample_method": "cv2.INTER_AREA",
                "difference_definition": "signed_difference = SR_down_crop - LR_crop; absolute_difference = abs(signed_difference)",
                "difference_name": "SR-to-LR reconstruction difference",
                "statistics": {k: row[k] for k in row if k.startswith(("mean_", "median_", "std_", "max_", "percentage_"))},
                "outputs": {
                    "sr_original": str(model_dir / "sr_original.png"),
                    "sr_downsampled_x4": str(model_dir / "sr_downsampled_x4.png"),
                    "lr_crop_512": str(model_dir / "lr_crop_512.png"),
                    "sr_downsampled_crop_512": str(model_dir / "sr_downsampled_crop_512.png"),
                    "difference_abs": str(model_dir / "difference_abs.png"),
                    "difference_signed_visualization": str(model_dir / "difference_signed.png"),
                    "difference_abs_raw": str(model_dir / "difference_abs_raw.npy"),
                    "difference_signed_raw": str(model_dir / "difference_signed_raw.npy"),
                    "comparison": str(model_dir / "comparison.png"),
                },
            }
            image_metadata["models"][model] = model_meta
            print(f"  {model} SR path: {sr_path}")
            print(f"  {model} SR dimensions: {sr.width}x{sr.height}")
            print(f"  {model} downsampled SR dimensions: {sr_down.width}x{sr_down.height}")
        (image_dir / "metadata.json").write_text(json.dumps(image_metadata, indent=2) + "\n")

    write_summary(out_dir / "summary.csv", summary_rows)
    final_report = {
        "task": "Reverse diagnostic: original LR vs SR downsampled back to LR",
        "output_dir": str(out_dir),
        "num_source_images": len(CASES),
        "models_per_image": ["swinsr", "hat"],
        "crop_size": CROP_SIZE,
        "scale": SCALE,
        "base_seed": SEED,
        "summary_csv": str(out_dir / "summary.csv"),
        "interpretation_warning": "This is not SR vs HR. Difference maps are SR-to-LR reconstruction differences and do not automatically prove artifacts.",
        "original_files_unchanged": {key: unchanged(state) for key, state in input_states.items()},
    }
    (out_dir / "README_reverse_diagnostic.txt").write_text(
        "\n".join(
            [
                "Reverse diagnostic: SR downsampled to LR vs original LR",
                "",
                "Comparison: original LR crop vs SR_downsampled_x4 crop.",
                "Difference name: SR-to-LR reconstruction difference.",
                "This is not an artifact map by itself and is not SR-vs-HR.",
                "",
                f"Output directory: {out_dir}",
                f"Summary CSV: {out_dir / 'summary.csv'}",
                f"Source images: {len(CASES)}",
                "Models per image: swinsr, hat",
                f"Crop size: {CROP_SIZE}x{CROP_SIZE}",
                f"Base seed: {SEED}; image_1 uses 42, image_2 uses 43.",
                "Downsampling method: cv2.INTER_AREA.",
                "Signed difference: SR_down_crop - LR_crop.",
                "Absolute difference: abs(SR_down_crop - LR_crop).",
                "Percentage thresholds use per-pixel max RGB absolute difference.",
                "",
                f"Original files unchanged: {all(final_report['original_files_unchanged'].values())}",
            ]
        )
        + "\n"
    )
    (out_dir / "run_metadata.json").write_text(json.dumps(final_report, indent=2) + "\n")

    print("\nConcise summary")
    print(f"Output directory: {out_dir}")
    for row in summary_rows:
        print(
            f"{row['image_folder']} {row['model']}: mean_abs={row['mean_absolute_difference']:.4f}, "
            f"max_abs={row['max_absolute_difference']}, pct_gt5={row['percentage_pixels_absdiff_gt_5']:.2f}%"
        )
    print(f"Original files unchanged: {all(final_report['original_files_unchanged'].values())}")


if __name__ == "__main__":
    main()
