#!/usr/bin/env python3
"""Check A: clean HR-calibration images vs heavy-blur degraded copies."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))

from dataloader import _resize_short_side, center_crop  # noqa: E402
from infer import load_best  # noqa: E402
from score import score_sz_eval  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def pil_to_tensor(image, device):
    arr = (np.asarray(image) / 255.0).astype("float32")
    return transforms.ToTensor()(arr).unsqueeze(0).to(device)


def save_pair(clean, degraded, path):
    out = Image.new("RGB", (clean.width + degraded.width, clean.height))
    out.paste(clean, (0, 0))
    out.paste(degraded, (clean.width, 0))
    out.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="/data/projectwork/swati_mam/HR_DATA/HR-calibration/DIV2K",
        help="Folder containing clean HR calibration images.",
    )
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"),
        help="Checkpoint used for S_Z / D scoring.",
    )
    parser.add_argument("--out_dir", default=str(ROOT / "task wacv"))
    parser.add_argument("--blur_radius", type=float, default=8.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    clean_dir = out_dir / "clean_scored_crops"
    deg_dir = out_dir / "degraded_heavy_blur"
    pair_dir = out_dir / "side_by_side"
    for d in (clean_dir, deg_dir, pair_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")

    generator, mu_ref, Sigma_ref, img_size, ldim, cfg = load_best(args.ckpt, device)
    mu_only = cfg.get("sz_mode", "mu_only") == "mu_only"
    sigma_t_max = cfg.get("sz_sigma_t_max", 1.0)

    paths = collect_images(args.data)
    if not paths:
        raise SystemExit(f"No images found under {args.data}")

    rows = []
    manifest_path = out_dir / "calibration_manifest.txt"
    manifest_path.write_text("\n".join(str(p) for p in paths) + "\n")

    with torch.no_grad():
        for idx, path in enumerate(paths, start=1):
            image = Image.open(path).convert("RGB")
            clean = center_crop(_resize_short_side(image, img_size), img_size)
            degraded = clean.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))

            clean_name = f"{idx:04d}_{path.stem}_clean.png"
            deg_name = f"{idx:04d}_{path.stem}_heavy_blur_r{args.blur_radius:g}.png"
            pair_name = f"{idx:04d}_{path.stem}_clean_vs_blur.png"

            clean_path = clean_dir / clean_name
            deg_path = deg_dir / deg_name
            pair_path = pair_dir / pair_name
            clean.save(clean_path)
            degraded.save(deg_path)
            save_pair(clean, degraded, pair_path)

            x_clean = pil_to_tensor(clean, device)
            x_deg = pil_to_tensor(degraded, device)
            d_clean = float(
                score_sz_eval(
                    generator,
                    x_clean,
                    mu_ref,
                    Sigma_ref,
                    sigma_t_max=sigma_t_max,
                    mu_only=mu_only,
                ).view(-1)[0].item()
            )
            d_deg = float(
                score_sz_eval(
                    generator,
                    x_deg,
                    mu_ref,
                    Sigma_ref,
                    sigma_t_max=sigma_t_max,
                    mu_only=mu_only,
                ).view(-1)[0].item()
            )
            diff = d_deg - d_clean
            rows.append(
                {
                    "index": idx,
                    "name": path.name,
                    "source_path": str(path),
                    "clean_saved": str(clean_path),
                    "degraded_saved": str(deg_path),
                    "pair_saved": str(pair_path),
                    "D_clean": d_clean,
                    "D_deg": d_deg,
                    "D_deg_minus_D_clean": diff,
                    "D_deg_gt_D_clean": int(diff > 0),
                }
            )

    csv_path = out_dir / "check_a_clean_vs_heavy_blur_scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    clean_scores = np.array([r["D_clean"] for r in rows], dtype=np.float64)
    deg_scores = np.array([r["D_deg"] for r in rows], dtype=np.float64)
    diffs = deg_scores - clean_scores
    paired_fraction = float(np.mean(diffs > 0))
    summary = {
        "check": "Check A - clean vs heavy blur degraded, calibration only",
        "data": args.data,
        "checkpoint": args.ckpt,
        "num_images": len(rows),
        "blur_radius": args.blur_radius,
        "score_name": "D = S_Z; higher means worse quality",
        "mean_D_clean": float(clean_scores.mean()),
        "mean_D_deg": float(deg_scores.mean()),
        "mean_difference_D_deg_minus_D_clean": float(diffs.mean()),
        "median_difference_D_deg_minus_D_clean": float(np.median(diffs)),
        "paired_fraction_D_deg_gt_D_clean": paired_fraction,
        "num_pairs_D_deg_gt_D_clean": int(np.sum(diffs > 0)),
        "passes_most_images_rule": bool(paired_fraction > 0.5),
        "scores_csv": str(csv_path),
        "manifest": str(manifest_path),
        "clean_crops_dir": str(clean_dir),
        "degraded_dir": str(deg_dir),
        "side_by_side_dir": str(pair_dir),
    }

    summary_json = out_dir / "check_a_summary.json"
    summary_txt = out_dir / "check_a_summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    summary_txt.write_text(
        "\n".join(
            [
                "Check A - clean vs degraded (calibration only)",
                f"Data: {summary['data']}",
                f"Checkpoint: {summary['checkpoint']}",
                f"Images: {summary['num_images']}",
                f"Degradation: Gaussian heavy blur, radius={summary['blur_radius']}",
                "Score: D = S_Z; higher means worse quality",
                f"mean D_clean: {summary['mean_D_clean']:.6f}",
                f"mean D_deg: {summary['mean_D_deg']:.6f}",
                "mean difference D_deg - D_clean: "
                f"{summary['mean_difference_D_deg_minus_D_clean']:.6f}",
                "median difference D_deg - D_clean: "
                f"{summary['median_difference_D_deg_minus_D_clean']:.6f}",
                "paired fraction D_deg > D_clean: "
                f"{summary['paired_fraction_D_deg_gt_D_clean']:.4f} "
                f"({summary['num_pairs_D_deg_gt_D_clean']}/{summary['num_images']})",
                f"passes most-images rule: {summary['passes_most_images_rule']}",
                f"Scores CSV: {summary['scores_csv']}",
            ]
        )
        + "\n"
    )

    print(summary_txt.read_text())


if __name__ == "__main__":
    main()
