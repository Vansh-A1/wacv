#!/usr/bin/env python3
"""Check B: severity trend for Gaussian blur on HR-calibration images."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))

from dataloader import _resize_short_side, center_crop  # noqa: E402
from infer import load_best  # noqa: E402
from score import score_sz_eval  # noqa: E402


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
BLUR_LADDER = [
    {"severity": 0, "kernel": 0, "sigma": 0.0},
    {"severity": 1, "kernel": 5, "sigma": 1.0},
    {"severity": 2, "kernel": 9, "sigma": 2.0},
    {"severity": 3, "kernel": 15, "sigma": 4.0},
    {"severity": 4, "kernel": 21, "sigma": 8.0},
]


def collect_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def apply_blur(image, kernel, sigma):
    if kernel == 0:
        return image.copy()
    arr = np.asarray(image)
    blurred = cv2.GaussianBlur(arr, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(blurred)


def pil_to_tensor(image, device):
    arr = (np.asarray(image) / 255.0).astype("float32")
    return transforms.ToTensor()(arr).unsqueeze(0).to(device)


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
        help="Checkpoint used for D_WACV = S_Z scoring.",
    )
    parser.add_argument("--out_dir", default=str(ROOT / "task wacv"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    severity_dir = out_dir / "check_b_gaussian_blur_severity_images"
    severity_dir.mkdir(parents=True, exist_ok=True)
    for level in BLUR_LADDER:
        (severity_dir / f"severity_{level['severity']}").mkdir(exist_ok=True)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")

    generator, mu_ref, Sigma_ref, img_size, _, cfg = load_best(args.ckpt, device)
    mu_only = cfg.get("sz_mode", "mu_only") == "mu_only"
    sigma_t_max = cfg.get("sz_sigma_t_max", 1.0)

    paths = collect_images(args.data)
    if not paths:
        raise SystemExit(f"No images found under {args.data}")

    rows = []
    by_image = {}
    with torch.no_grad():
        for idx, path in enumerate(paths, start=1):
            image = Image.open(path).convert("RGB")
            clean_crop = center_crop(_resize_short_side(image, img_size), img_size)
            image_scores = []

            for level in BLUR_LADDER:
                severity = level["severity"]
                kernel = level["kernel"]
                sigma = level["sigma"]
                degraded = apply_blur(clean_crop, kernel, sigma)
                saved_name = f"{idx:04d}_{path.stem}_sev{severity}_k{kernel}_sigma{sigma:g}.png"
                saved_path = severity_dir / f"severity_{severity}" / saved_name
                degraded.save(saved_path)

                x = pil_to_tensor(degraded, device)
                score = float(
                    score_sz_eval(
                        generator,
                        x,
                        mu_ref,
                        Sigma_ref,
                        sigma_t_max=sigma_t_max,
                        mu_only=mu_only,
                    ).view(-1)[0].item()
                )
                image_scores.append(score)
                rows.append(
                    {
                        "index": idx,
                        "name": path.name,
                        "source_path": str(path),
                        "severity": severity,
                        "kernel": kernel,
                        "sigma": sigma,
                        "D_WACV": score,
                        "saved_image": str(saved_path),
                    }
                )

            by_image[path.name] = image_scores

    csv_path = out_dir / "check_b_gaussian_blur_severity_scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pooled_severities = np.array([r["severity"] for r in rows], dtype=np.float64)
    pooled_scores = np.array([r["D_WACV"] for r in rows], dtype=np.float64)
    pooled_spearman = spearmanr(pooled_severities, pooled_scores)

    per_image = []
    adjacent_total = 0
    adjacent_correct = 0
    strict_full_order = 0
    for name, scores in by_image.items():
        severities = np.arange(len(scores), dtype=np.float64)
        rho = spearmanr(severities, np.array(scores, dtype=np.float64)).statistic
        adjacent_hits = sum(1 for a, b in zip(scores, scores[1:]) if b > a)
        adjacent_total += len(scores) - 1
        adjacent_correct += adjacent_hits
        if adjacent_hits == len(scores) - 1:
            strict_full_order += 1
        per_image.append(
            {
                "name": name,
                "spearman_severity_D_WACV": float(rho),
                "adjacent_correct": adjacent_hits,
                "adjacent_total": len(scores) - 1,
                "strictly_increasing_all_levels": int(adjacent_hits == len(scores) - 1),
                "scores_by_severity": scores,
            }
        )

    per_image_csv = out_dir / "check_b_per_image_trend.csv"
    with per_image_csv.open("w", newline="") as f:
        fieldnames = [
            "name",
            "spearman_severity_D_WACV",
            "adjacent_correct",
            "adjacent_total",
            "strictly_increasing_all_levels",
            "scores_by_severity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image)

    adjacent_rate = adjacent_correct / adjacent_total
    positive_rhos = [r["spearman_severity_D_WACV"] for r in per_image if r["spearman_severity_D_WACV"] > 0]
    summary = {
        "check": "Check B - Gaussian blur severity trend, calibration only",
        "data": args.data,
        "checkpoint": args.ckpt,
        "num_images": len(paths),
        "severity_ladder": BLUR_LADDER,
        "score_name": "D_WACV = S_Z; higher means worse quality",
        "pooled_spearman_rho_severity_D_WACV": float(pooled_spearman.statistic),
        "pooled_spearman_pvalue": float(pooled_spearman.pvalue),
        "adjacent_ordering_rate": float(adjacent_rate),
        "adjacent_pairs_correct": int(adjacent_correct),
        "adjacent_pairs_total": int(adjacent_total),
        "per_image_positive_spearman_fraction": float(len(positive_rhos) / len(per_image)),
        "per_image_strictly_increasing_fraction": float(strict_full_order / len(per_image)),
        "num_images_strictly_increasing": int(strict_full_order),
        "scores_csv": str(csv_path),
        "per_image_csv": str(per_image_csv),
        "severity_images_dir": str(severity_dir),
        "passes_positive_spearman_rule": bool(pooled_spearman.statistic > 0),
    }

    summary_json = out_dir / "check_b_summary.json"
    summary_txt = out_dir / "check_b_summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    summary_txt.write_text(
        "\n".join(
            [
                "Check B - severity trend (calibration only)",
                f"Data: {summary['data']}",
                f"Checkpoint: {summary['checkpoint']}",
                f"Images: {summary['num_images']}",
                "Degradation: cv2.GaussianBlur",
                f"Severity ladder: {summary['severity_ladder']}",
                "Score: D_WACV = S_Z; higher means worse quality",
                "pooled Spearman(severity, D_WACV): "
                f"{summary['pooled_spearman_rho_severity_D_WACV']:.6f}",
                f"pooled Spearman p-value: {summary['pooled_spearman_pvalue']:.6g}",
                "adjacent ordering rate D(i+1) > D(i): "
                f"{summary['adjacent_ordering_rate']:.4f} "
                f"({summary['adjacent_pairs_correct']}/{summary['adjacent_pairs_total']})",
                "per-image positive Spearman fraction: "
                f"{summary['per_image_positive_spearman_fraction']:.4f}",
                "per-image strictly increasing fraction: "
                f"{summary['per_image_strictly_increasing_fraction']:.4f} "
                f"({summary['num_images_strictly_increasing']}/{summary['num_images']})",
                f"passes positive Spearman rule: {summary['passes_positive_spearman_rule']}",
                f"Scores CSV: {summary['scores_csv']}",
                f"Per-image CSV: {summary['per_image_csv']}",
            ]
        )
        + "\n"
    )

    print(summary_txt.read_text())


if __name__ == "__main__":
    main()
