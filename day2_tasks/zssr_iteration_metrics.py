#!/usr/bin/env python3
"""Compute ZSSR per-iteration metric trends for the six vansh_zssr folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-zssr-metrics")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))

from dataloader import _resize_short_side, center_crop  # noqa: E402
from infer import load_best  # noqa: E402
from score import score_sz_eval  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_ZSSR_ROOT = Path("/home/projectwork/vansh_zssr")
DEFAULT_LR_DIR = Path(
    "/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/"
    "DRealSR-20231109T095957Z-004/DRealSR/Test_x4/LR"
)
DEFAULT_HR_DIR = Path(
    "/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/"
    "DRealSR-20231109T095957Z-004/DRealSR/Test_x4/HR"
)
DEFAULT_CKPT = ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"
CSV_COLUMNS = [
    "folder",
    "image_id",
    "iteration",
    "iteration_path",
    "hr_path",
    "lr_path",
    "D_WACV",
    "NIQE",
    "DBCNN",
    "LR_content",
    "PSNR",
    "LPIPS",
]


def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def strip_hr_id(path: Path) -> str:
    return re.sub(r"_x4$", "", path.stem)


def lr_number(path: Path) -> int | None:
    match = re.search(r"dreal4_(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def build_lr_map(lr_dir: Path, hr_dir: Path) -> dict[str, Path]:
    hr_ids = [strip_hr_id(p) for p in sorted(collect_images(hr_dir), key=lambda x: x.name)]
    numbered = sorted([p for p in collect_images(lr_dir) if lr_number(p) is not None], key=lr_number)
    if len(hr_ids) != len(numbered):
        raise RuntimeError(f"Cannot map LR to HR ids: {len(numbered)} LR files, {len(hr_ids)} HR files.")
    return dict(zip(hr_ids, numbered))


def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = (np.asarray(image).astype(np.float32) / 255.0).clip(0.0, 1.0)
    return transforms.ToTensor()(arr).unsqueeze(0).to(device)


def center_metric_crop(image: Image.Image, crop_size: int) -> Image.Image:
    if crop_size <= 0:
        return image
    if image.width < crop_size or image.height < crop_size:
        image = _resize_short_side(image, crop_size)
    return center_crop(image, crop_size)


def score_wacv(generator, image: Image.Image, mu_ref, sigma_ref, img_size: int, cfg: dict, device: torch.device) -> float:
    crop = center_crop(_resize_short_side(image, img_size), img_size)
    x = pil_to_tensor(crop, device)
    score = score_sz_eval(
        generator,
        x,
        mu_ref,
        sigma_ref,
        sigma_t_max=cfg.get("sz_sigma_t_max", 1.0),
        mu_only=(cfg.get("sz_mode", "mu_only") == "mu_only"),
    )
    return float(score.view(-1)[0].item())


def psnr_rgb(sr_crop: Image.Image, hr_crop: Image.Image) -> float:
    sr = np.asarray(sr_crop).astype(np.float32) / 255.0
    hr = np.asarray(hr_crop).astype(np.float32) / 255.0
    mse = float(np.mean((sr - hr) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def lr_content(sr_image: Image.Image, lr_image: Image.Image) -> float:
    down = sr_image.resize((sr_image.width // 4, sr_image.height // 4), Image.Resampling.BICUBIC)
    if down.size != lr_image.size:
        raise ValueError(f"Down4(SR) size {down.size} does not match LR size {lr_image.size}")
    down_arr = np.asarray(down).astype(np.float32) / 255.0
    lr_arr = np.asarray(lr_image).astype(np.float32) / 255.0
    return float(np.sqrt(np.mean((down_arr - lr_arr) ** 2)))


def iteration_number(path: Path) -> int:
    match = re.search(r"iteration_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse iteration number from {path}")
    return int(match.group(1))


def find_zssr_folders(zssr_root: Path) -> list[Path]:
    folders = []
    for folder in sorted(p for p in zssr_root.iterdir() if p.is_dir()):
        iter_dir = folder / "intermediate" / "X4.00X4.00"
        hr_dir = folder / "intermediate" / "hr"
        if iter_dir.is_dir() and hr_dir.is_dir():
            folders.append(folder)
    return folders


def plot_group(rows: list[dict[str, object]], metrics: list[str], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 8), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    xs = [int(r["iteration"]) for r in rows]
    for ax, metric in zip(axes, metrics):
        ys = [float(r[metric]) for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.7)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("iteration")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zssr_root", default=str(DEFAULT_ZSSR_ROOT))
    parser.add_argument("--lr_dir", default=str(DEFAULT_LR_DIR))
    parser.add_argument("--hr_dir", default=str(DEFAULT_HR_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_ZSSR_ROOT / "zssr_metric_trends"))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--metric_crop", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    import pyiqa

    zssr_root = Path(args.zssr_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_folder_dir = out_dir / "per_folder"
    plot_dir = out_dir / "plots"
    per_folder_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")

    generator, mu_ref, sigma_ref, img_size, ldim, cfg = load_best(args.ckpt, device)
    generator.eval()

    pyiqa_metrics = {
        "niqe": pyiqa.create_metric("niqe", device=device),
        "lpips": pyiqa.create_metric("lpips", device=device),
        "dbcnn": pyiqa.create_metric("dbcnn", device=device),
    }

    lr_map = build_lr_map(Path(args.lr_dir), Path(args.hr_dir))
    folders = find_zssr_folders(zssr_root)
    if not folders:
        raise SystemExit(f"No ZSSR folders found under {zssr_root}")

    all_rows: list[dict[str, object]] = []
    folder_summaries: list[dict[str, object]] = []
    with torch.no_grad():
        for folder in folders:
            image_id = re.sub(r"_x1$", "", folder.name)
            iter_dir = folder / "intermediate" / "X4.00X4.00"
            local_hrs = collect_images(folder / "intermediate" / "hr")
            if len(local_hrs) != 1:
                raise RuntimeError(f"Expected one HR image in {folder / 'intermediate' / 'hr'}, found {len(local_hrs)}")
            hr_path = local_hrs[0]
            lr_path = lr_map.get(image_id)
            if lr_path is None:
                raise RuntimeError(f"No matched LR path found for {image_id}")

            hr_image = pil_rgb(hr_path)
            lr_image = pil_rgb(lr_path)
            iter_paths = sorted(collect_images(iter_dir), key=iteration_number)
            folder_rows: list[dict[str, object]] = []
            for idx, iter_path in enumerate(iter_paths, start=1):
                sr_image = pil_rgb(iter_path)
                if sr_image.size != hr_image.size:
                    raise RuntimeError(f"Size mismatch for {iter_path}: SR {sr_image.size}, HR {hr_image.size}")
                if sr_image.size != (lr_image.width * 4, lr_image.height * 4):
                    raise RuntimeError(f"Scale mismatch for {iter_path}: SR {sr_image.size}, LR {lr_image.size}")

                sr_crop = center_metric_crop(sr_image, args.metric_crop)
                hr_crop = center_metric_crop(hr_image, args.metric_crop)
                sr_t = pil_to_tensor(sr_crop, device)
                hr_t = pil_to_tensor(hr_crop, device)

                row = {
                    "folder": folder.name,
                    "image_id": image_id,
                    "iteration": iteration_number(iter_path),
                    "iteration_path": str(iter_path),
                    "hr_path": str(hr_path),
                    "lr_path": str(lr_path),
                    "D_WACV": score_wacv(generator, sr_image, mu_ref, sigma_ref, img_size, cfg, device),
                    "NIQE": float(pyiqa_metrics["niqe"](sr_t).view(-1)[0].item()),
                    "DBCNN": float(pyiqa_metrics["dbcnn"](sr_t).view(-1)[0].item()),
                    "LR_content": lr_content(sr_image, lr_image),
                    "PSNR": psnr_rgb(sr_crop, hr_crop),
                    "LPIPS": float(pyiqa_metrics["lpips"](sr_t, hr_t).view(-1)[0].item()),
                }
                folder_rows.append(row)
                all_rows.append(row)
                print(f"[{folder.name} {idx}/{len(iter_paths)}] iteration {row['iteration']}", end="\r")
            print()

            write_csv(per_folder_dir / f"{folder.name}_iteration_metrics.csv", folder_rows)
            plot_group(
                folder_rows,
                ["D_WACV", "NIQE", "DBCNN"],
                plot_dir / f"{folder.name}_wacv_niqe_dbcnn_vs_iter.png",
                f"{folder.name}: D_WACV, NIQE, DBCNN vs iteration",
            )
            plot_group(
                folder_rows,
                ["LR_content", "PSNR", "LPIPS"],
                plot_dir / f"{folder.name}_lrcontent_psnr_lpips_vs_iter.png",
                f"{folder.name}: LR_content, PSNR, LPIPS vs iteration",
            )
            folder_summaries.append(
                {
                    "folder": folder.name,
                    "image_id": image_id,
                    "num_iterations": len(folder_rows),
                    "first_iteration": int(folder_rows[0]["iteration"]),
                    "last_iteration": int(folder_rows[-1]["iteration"]),
                    "csv": str(per_folder_dir / f"{folder.name}_iteration_metrics.csv"),
                    "plot_quality": str(plot_dir / f"{folder.name}_wacv_niqe_dbcnn_vs_iter.png"),
                    "plot_reference": str(plot_dir / f"{folder.name}_lrcontent_psnr_lpips_vs_iter.png"),
                }
            )

    write_csv(out_dir / "zssr_all_iteration_metrics.csv", all_rows)

    summary = {
        "task": "ZSSR iteration metric trends",
        "zssr_root": str(zssr_root),
        "output_dir": str(out_dir),
        "num_folders": len(folders),
        "total_rows": len(all_rows),
        "checkpoint": args.ckpt,
        "wacv_score": "D_WACV = S_Z; higher means worse quality, same frozen checkpoint/wrapper direction as previous tasks.",
        "metric_crop": args.metric_crop,
        "metric_note": "D_WACV, NIQE, DBCNN, PSNR, and LPIPS were computed on the same 256x256 center crop for reproducibility and speed. LR_content was computed on the full iteration image after bicubic Down4(SR) against matched LR.",
        "pyiqa_version": getattr(pyiqa, "__version__", "unknown"),
        "image_size": int(img_size),
        "latent_dim": int(ldim),
        "folders": folder_summaries,
        "outputs": {
            "all_metrics_csv": str(out_dir / "zssr_all_iteration_metrics.csv"),
            "per_folder_dir": str(per_folder_dir),
            "plot_dir": str(plot_dir),
            "summary_json": str(out_dir / "zssr_metric_trends_summary.json"),
            "summary_txt": str(out_dir / "zssr_metric_trends_summary.txt"),
        },
    }
    (out_dir / "zssr_metric_trends_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "ZSSR iteration metric trends",
        f"ZSSR root: {zssr_root}",
        f"Output dir: {out_dir}",
        f"Folders scored: {len(folders)}",
        f"Total rows: {len(all_rows)}",
        f"Metric crop: {args.metric_crop}x{args.metric_crop}",
        "D_WACV, NIQE, DBCNN vs iter plots and LR_content, PSNR, LPIPS vs iter plots are saved in plots/.",
        "D_WACV uses the frozen WACV checkpoint and score direction from previous tasks.",
        "LR_content uses full-image bicubic Down4(SR) vs matched LR; PSNR/LPIPS use matched HR center crop.",
        "",
        "Per-folder outputs:",
    ]
    for item in folder_summaries:
        lines.append(f"{item['folder']}: {item['num_iterations']} iterations, CSV={item['csv']}")
    (out_dir / "zssr_metric_trends_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote outputs to {out_dir}")
    print(f"Rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
