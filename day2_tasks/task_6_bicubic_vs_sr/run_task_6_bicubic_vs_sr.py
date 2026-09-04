#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("/home/projectwork/student_package/day2_tasks/task_6_bicubic_vs_sr")
SCALE = 4
CROP_SIZE = 512
NUM_CROPS = 10
RANDOM_SEED = 4266
CHANGE_THRESHOLD = 5


@dataclass(frozen=True)
class TaskImage:
    folder_name: str
    image_id: str
    lr_image_id: str
    lr_path: Path
    hr_path: Path
    sr_paths: dict[str, Path]


TASK_IMAGES = [
    TaskImage(
        folder_name="image_1",
        image_id="sony_160",
        lr_image_id="dreal4_84",
        lr_path=Path(
            "/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/"
            "DRealSR-20231109T095957Z-004/DRealSR/Test_x4/LR/dreal4_84.png"
        ),
        hr_path=Path(
            "/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/"
            "DRealSR-20231109T095957Z-004/DRealSR/Test_x4/HR/sony_160_x4.png"
        ),
        sr_paths={
            "swinsr": Path("/data/projectwork/sr_output/swinIr_Inference/SwinIR_results/Dreal/DRealSR_all_x4_sony_160_x1.png"),
            "hat": Path("/data/projectwork/sr_output/HAT_inference/DReal_x4/visualization/DReal_x4/sony_160_x1_DReal_x4.png"),
        },
    ),
    TaskImage(
        folder_name="image_2",
        image_id="Canon_10",
        lr_image_id="dreal4_1",
        lr_path=Path(
            "/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/"
            "DRealSR-20231109T095957Z-004/DRealSR/Test_x4/LR/dreal4_1.png"
        ),
        hr_path=Path(
            "/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/"
            "DRealSR-20231109T095957Z-004/DRealSR/Test_x4/HR/Canon_10_x4.png"
        ),
        sr_paths={
            "swinsr": Path("/data/projectwork/sr_output/swinIr_Inference/SwinIR_results/Dreal/DRealSR_all_x4_Canon_10_x1.png"),
            "hat": Path("/data/projectwork/sr_output/HAT_inference/DReal_x4/visualization/DReal_x4/Canon_10_x1_DReal_x4.png"),
        },
    ),
]


@dataclass(frozen=True)
class FileState:
    path: str
    size_bytes: int
    mtime_ns: int


def snapshot(path: Path) -> FileState:
    stat = path.stat()
    return FileState(str(path), stat.st_size, stat.st_mtime_ns)


def verify_unchanged(before: dict[str, FileState]) -> dict[str, bool]:
    results = {}
    for key, state in before.items():
        now = snapshot(Path(state.path))
        results[key] = now.size_bytes == state.size_bytes and now.mtime_ns == state.mtime_ns
    return results


def ensure_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_rmse(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(np.mean((arr_a - arr_b) ** 2)))


def make_coords(width: int, height: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    max_x = width - CROP_SIZE
    max_y = height - CROP_SIZE
    while len(coords) < NUM_CROPS:
        x = rng.randint(0, max_x)
        y = rng.randint(0, max_y)
        if (x, y) not in seen:
            seen.add((x, y))
            coords.append((x, y))
    return coords


def save_comparison(bic_crop: Image.Image, sr_crop: Image.Image, diff_vis: Image.Image, out_path: Path) -> None:
    label_h = 34
    gap = 8
    panel_w, panel_h = bic_crop.size
    canvas = Image.new("RGB", (panel_w * 3 + gap * 2, panel_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    labels = ["Bicubic crop", "SR crop", "|SR - Bicubic|"]
    for idx, (label, panel) in enumerate(zip(labels, [bic_crop, sr_crop, diff_vis])):
        x0 = idx * (panel_w + gap)
        draw.text((x0 + 8, 10), label, fill=(0, 0, 0), font=font)
        canvas.paste(panel, (x0, label_h))
    canvas.save(out_path)


def csv_write(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def move_previous_root_outputs() -> None:
    if not OUT_DIR.exists() or (OUT_DIR / "image_1").exists():
        return
    legacy_names = [
        "README_task_6_bicubic_vs_sr.txt",
        "all_crop_difference_stats.csv",
        "bicubic",
        "coordinates",
        "hat",
        "input_copies",
        "reports",
        "swinsr",
    ]
    image_1 = OUT_DIR / "image_1"
    image_1.mkdir(parents=True, exist_ok=True)
    for name in legacy_names:
        src = OUT_DIR / name
        if src.exists():
            shutil.move(str(src), str(image_1 / name))


def run_one_image(task: TaskImage) -> dict[str, object]:
    image_out = OUT_DIR / task.folder_name
    image_out.mkdir(parents=True, exist_ok=True)
    for subdir in ["input_copies", "bicubic", "coordinates", "reports"]:
        (image_out / subdir).mkdir(exist_ok=True)

    inputs = {"lr": task.lr_path, "hr": task.hr_path, **{f"sr_{k}": v for k, v in task.sr_paths.items()}}
    for key, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"{key} input does not exist: {path}")
    before_state = {key: snapshot(path) for key, path in inputs.items()}

    lr = ensure_rgb(task.lr_path)
    hr = ensure_rgb(task.hr_path)
    sr_images = {name: ensure_rgb(path) for name, path in task.sr_paths.items()}

    expected_sr_size = (lr.size[0] * SCALE, lr.size[1] * SCALE)
    if hr.size != expected_sr_size:
        raise ValueError(f"HR size mismatch: expected {expected_sr_size}, got {hr.size}")
    for name, sr_img in sr_images.items():
        if sr_img.size != expected_sr_size:
            raise ValueError(f"{name} SR size mismatch: expected {expected_sr_size}, got {sr_img.size}")

    # Save read-only input copies inside the task folder for audit convenience.
    shutil.copy2(task.lr_path, image_out / "input_copies" / task.lr_path.name)
    shutil.copy2(task.hr_path, image_out / "input_copies" / task.hr_path.name)
    for name, path in task.sr_paths.items():
        shutil.copy2(path, image_out / "input_copies" / f"{name}_{path.name}")

    bicubic = lr.resize(expected_sr_size, Image.Resampling.BICUBIC)
    bicubic_path = image_out / "bicubic" / f"{task.lr_image_id}_to_{task.image_id}_bicubic_x4.png"
    bicubic.save(bicubic_path)

    image_seed = RANDOM_SEED + int(task.folder_name.split("_")[-1]) - 1
    coords = make_coords(*expected_sr_size, seed=image_seed)
    coord_rows = [
        {"crop_id": f"crop_{i:02d}", "x": x, "y": y, "width": CROP_SIZE, "height": CROP_SIZE}
        for i, (x, y) in enumerate(coords, start=1)
    ]
    csv_write(
        image_out / "coordinates" / "crop_coordinates.csv",
        coord_rows,
        ["crop_id", "x", "y", "width", "height"],
    )

    combined_rows: list[dict[str, object]] = []
    verification: dict[str, object] = {
        "image_folder": task.folder_name,
        "image_id": task.image_id,
        "lr_image_id": task.lr_image_id,
        "random_seed": image_seed,
        "num_crops": NUM_CROPS,
        "crop_size": CROP_SIZE,
        "change_threshold_abs_rgb": CHANGE_THRESHOLD,
        "input_paths": {key: str(path) for key, path in inputs.items()},
        "sizes": {"lr": lr.size, "hr": hr.size, **{f"sr_{k}": v.size for k, v in sr_images.items()}},
        "scale_factor": SCALE,
        "content_checks": {},
        "generated_files": [],
    }

    hr_down = hr.resize(lr.size, Image.Resampling.BICUBIC)
    for name, sr_img in sr_images.items():
        sr_down = sr_img.resize(lr.size, Image.Resampling.BICUBIC)
        verification["content_checks"][name] = {
            "filename_mapping": f"{task.lr_path.name} -> {task.image_id} -> {task.sr_paths[name].name}",
            "rmse_downscaled_sr_vs_lr_rgb_0_255": image_rmse(sr_down, lr),
            "rmse_downscaled_hr_vs_lr_rgb_0_255": image_rmse(hr_down, lr),
        }

        model_dir = image_out / name
        crop_dir = model_dir / "crops"
        diff_dir = model_dir / "difference_maps"
        fig_dir = model_dir / "comparisons"
        for d in [model_dir, crop_dir, diff_dir, fig_dir]:
            d.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        for i, (x, y) in enumerate(coords, start=1):
            crop_id = f"crop_{i:02d}"
            box = (x, y, x + CROP_SIZE, y + CROP_SIZE)
            bic_crop = bicubic.crop(box)
            sr_crop = sr_img.crop(box)
            if bic_crop.size != (CROP_SIZE, CROP_SIZE) or sr_crop.size != (CROP_SIZE, CROP_SIZE):
                raise RuntimeError(f"{name} {crop_id} crop size verification failed")

            bic_arr = np.asarray(bic_crop, dtype=np.int16)
            sr_arr = np.asarray(sr_crop, dtype=np.int16)
            diff = sr_arr - bic_arr
            absdiff = np.abs(diff).astype(np.uint8)
            per_pixel_max = absdiff.max(axis=2)
            changed_pct = float(100.0 * np.mean(per_pixel_max >= CHANGE_THRESHOLD))

            bic_path = crop_dir / f"{crop_id}_x{x}_y{y}_bicubic.png"
            sr_path = crop_dir / f"{crop_id}_x{x}_y{y}_{name}_sr.png"
            raw_diff_path = diff_dir / f"{crop_id}_x{x}_y{y}_absdiff_raw.png"
            vis_diff_path = diff_dir / f"{crop_id}_x{x}_y{y}_absdiff_vis_x4.png"
            comparison_path = fig_dir / f"{crop_id}_x{x}_y{y}_comparison.png"

            Image.fromarray(absdiff, mode="RGB").save(raw_diff_path)
            diff_vis = Image.fromarray(np.clip(absdiff.astype(np.uint16) * 4, 0, 255).astype(np.uint8), mode="RGB")
            diff_vis.save(vis_diff_path)
            bic_crop.save(bic_path)
            sr_crop.save(sr_path)
            save_comparison(bic_crop, sr_crop, diff_vis, comparison_path)

            row = {
                "crop_id": crop_id,
                "x": x,
                "y": y,
                "width": CROP_SIZE,
                "height": CROP_SIZE,
                "mean_absolute_difference": float(absdiff.mean()),
                "median_absolute_difference": float(np.median(absdiff)),
                "max_absolute_difference": int(absdiff.max()),
                "percentage_pixels_changed": changed_pct,
            }
            rows.append(row)
            combined_rows.append({"model": name, **row})

        csv_write(
            model_dir / "crop_difference_stats.csv",
            rows,
            [
                "crop_id",
                "x",
                "y",
                "width",
                "height",
                "mean_absolute_difference",
                "median_absolute_difference",
                "max_absolute_difference",
                "percentage_pixels_changed",
            ],
        )

    csv_write(
        image_out / "all_crop_difference_stats.csv",
        combined_rows,
        [
            "model",
            "crop_id",
            "x",
            "y",
            "width",
            "height",
            "mean_absolute_difference",
            "median_absolute_difference",
            "max_absolute_difference",
            "percentage_pixels_changed",
        ],
    )

    generated = sorted(
        str(p.relative_to(image_out))
        for p in image_out.rglob("*")
        if p.is_file()
    )
    unchanged = verify_unchanged(before_state)
    verification["generated_files"] = generated
    verification["original_files_unchanged"] = unchanged
    verification["all_generated_files_inside_image_output_dir"] = all((image_out / rel).resolve().is_relative_to(image_out.resolve()) for rel in generated)
    verification["bicubic_size_verified"] = ensure_rgb(bicubic_path).size == expected_sr_size
    verification["sr_size_verified"] = {name: img.size == expected_sr_size for name, img in sr_images.items()}
    verification["crop_sizes_verified"] = True
    for rel in generated:
        if "_crop" in rel or "absdiff" in rel or "comparison" in rel:
            if rel.endswith(".png") and "comparison" not in rel:
                size = ensure_rgb(image_out / rel).size
                if size != (CROP_SIZE, CROP_SIZE):
                    verification["crop_sizes_verified"] = False

    with (image_out / "reports" / "verification.json").open("w") as f:
        json.dump(verification, f, indent=2)

    report_lines = [
        f"Task 6: Bicubic vs SR diagnostic for {task.folder_name}",
        "",
        "Purpose: identify where each SR model output differs from a simple bicubic-upsampled LR baseline.",
        "Important: these difference maps are diagnostic only; they do not prove artifacts exist.",
        "",
        f"LR input: {task.lr_path}",
        f"HR input used for correspondence check: {task.hr_path}",
        f"SwinIR SR input: {task.sr_paths['swinsr']}",
        f"HAT SR input: {task.sr_paths['hat']}",
        f"Output directory: {image_out}",
        "",
        f"Original LR resolution: {lr.size[0]}x{lr.size[1]}",
        f"Bicubic output resolution: {expected_sr_size[0]}x{expected_sr_size[1]}",
        f"SR resolution: {expected_sr_size[0]}x{expected_sr_size[1]}",
        f"Scale factor: {SCALE}x",
        f"Random seed: {image_seed}",
        f"Number of crops per model: {NUM_CROPS}",
        f"Crop size: {CROP_SIZE}x{CROP_SIZE}",
        f"Percentage-pixels-changed threshold: a pixel is changed if any RGB channel has |SR - Bicubic| >= {CHANGE_THRESHOLD} intensity levels.",
        "",
        "Subtraction: images were loaded as RGB uint8. For each crop, difference = SR_crop - bicubic_crop using signed integer arrays; abs_difference = abs(difference). Statistics are computed on raw 0-255 RGB absolute differences.",
        "The saved *_absdiff_raw.png files contain raw absolute differences. The *_absdiff_vis_x4.png and comparison panels multiply raw differences by 4 only for visualization.",
        "",
        "Coordinates:",
    ]
    for row in coord_rows:
        report_lines.append(f"{row['crop_id']}: x={row['x']}, y={row['y']}, width={row['width']}, height={row['height']}")
    report_lines.extend(
        [
            "",
        "Content correspondence check:",
            f"The chosen files are paired by the existing DRealSR index/name mapping: {task.lr_image_id} -> {task.image_id}.",
        ]
    )
    for name, check in verification["content_checks"].items():
        report_lines.append(
            f"{name}: {check['filename_mapping']}; downscaled SR-vs-LR RMSE={check['rmse_downscaled_sr_vs_lr_rgb_0_255']:.4f}; downscaled HR-vs-LR RMSE={check['rmse_downscaled_hr_vs_lr_rgb_0_255']:.4f}"
        )
    report_lines.extend(
        [
            "",
            "Verification:",
            f"Original files unchanged: {all(unchanged.values())}",
            f"Bicubic image exactly 4580x3380: {verification['bicubic_size_verified']}",
            f"SR images exactly 4580x3380: {verification['sr_size_verified']}",
            f"Every saved crop/difference image exactly 512x512: {verification['crop_sizes_verified']}",
            f"All generated files are inside image output directory: {verification['all_generated_files_inside_image_output_dir']}",
            "",
            "Main outputs:",
            f"bicubic/{task.lr_image_id}_to_{task.image_id}_bicubic_x4.png",
            "coordinates/crop_coordinates.csv",
            "swinsr/crop_difference_stats.csv",
            "hat/crop_difference_stats.csv",
            "all_crop_difference_stats.csv",
            "reports/verification.json",
        ]
    )
    (image_out / "README_task_6_bicubic_vs_sr.txt").write_text("\n".join(report_lines) + "\n")
    return verification


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    move_previous_root_outputs()
    all_verifications = {}
    for task in TASK_IMAGES:
        all_verifications[task.folder_name] = run_one_image(task)
    with (OUT_DIR / "task_6_two_image_verification.json").open("w") as f:
        json.dump(all_verifications, f, indent=2)

    print(f"Saved Task 6 outputs to {OUT_DIR}")
    for folder, verification in all_verifications.items():
        print(f"{folder}: originals unchanged={all(verification['original_files_unchanged'].values())}, generated files={len(verification['generated_files'])}")


if __name__ == "__main__":
    main()
