#!/usr/bin/env python3
"""Task B / Check D: content sensitivity on HR-calibration images."""

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
BLUR_LEVEL_4 = {"severity": 4, "kernel": 21, "sigma": 8.0}


def collect_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def pil_to_float_rgb(image):
    return (np.asarray(image).astype(np.float32) / 255.0).clip(0.0, 1.0)


def pil_to_tensor(image, device):
    arr = pil_to_float_rgb(image)
    return transforms.ToTensor()(arr).unsqueeze(0).to(device)


def luminance(rgb):
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def content_proxies(clean_crop):
    rgb = pil_to_float_rgb(clean_crop)
    y = luminance(rgb).astype(np.float32)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    return float(y.mean()), float(grad_mag.mean())


def apply_blur_level_4(image):
    arr = np.asarray(image)
    blurred = cv2.GaussianBlur(
        arr,
        (BLUR_LEVEL_4["kernel"], BLUR_LEVEL_4["kernel"]),
        sigmaX=BLUR_LEVEL_4["sigma"],
        sigmaY=BLUR_LEVEL_4["sigma"],
    )
    return Image.fromarray(blurred)


def score_image(generator, image, mu_ref, Sigma_ref, cfg, device):
    x = pil_to_tensor(image, device)
    score = score_sz_eval(
        generator,
        x,
        mu_ref,
        Sigma_ref,
        sigma_t_max=cfg.get("sz_sigma_t_max", 1.0),
        mu_only=(cfg.get("sz_mode", "mu_only") == "mu_only"),
    )
    return float(score.view(-1)[0].item())


def report_row(quantity, value):
    return {"quantity": quantity, "value": value}


def write_report_txt(path, report_rows):
    lines = [
        "Task B - Content sensitivity test (Check D)",
        "Score: D_WACV = S_Z; higher means worse quality",
        "Blur level 4: kernel=21, sigma=8.0",
        "",
        "quantity,value",
    ]
    for row in report_rows:
        value = row["value"]
        if isinstance(value, float):
            value = f"{value:.6f}"
        lines.append(f"{row['quantity']},{value}")
    path.write_text("\n".join(lines) + "\n")


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
    parser.add_argument("--out_dir", default=str(ROOT / "day2_tasks"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    clean_dir = out_dir / "check_d_clean_crops"
    blur_dir = out_dir / "check_d_blur_level4_images"
    clean_dir.mkdir(parents=True, exist_ok=True)
    blur_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")

    generator, mu_ref, Sigma_ref, img_size, ldim, cfg = load_best(args.ckpt, device)
    paths = collect_images(args.data)
    if not paths:
        raise SystemExit(f"No images found under {args.data}")

    rows = []
    with torch.no_grad():
        for idx, path in enumerate(paths, start=1):
            image = Image.open(path).convert("RGB")
            clean = center_crop(_resize_short_side(image, img_size), img_size)
            blur4 = apply_blur_level_4(clean)

            clean_saved = clean_dir / f"{idx:04d}_{path.stem}_clean.png"
            blur_saved = blur_dir / f"{idx:04d}_{path.stem}_blur4_k21_sigma8.png"
            clean.save(clean_saved)
            blur4.save(blur_saved)

            d0 = score_image(generator, clean, mu_ref, Sigma_ref, cfg, device)
            d4 = score_image(generator, blur4, mu_ref, Sigma_ref, cfg, device)
            lum, mean_grad = content_proxies(clean)
            delta = d4 - d0

            rows.append(
                {
                    "index": idx,
                    "image_id": path.name,
                    "source_path": str(path),
                    "clean_crop": str(clean_saved),
                    "blur4_image": str(blur_saved),
                    "D0": d0,
                    "D4": d4,
                    "delta": delta,
                    "delta_gt_0": int(delta > 0),
                    "luminance": lum,
                    "mean_gradient_magnitude": mean_grad,
                    "edge_density": mean_grad,
                }
            )
            print(f"[{idx}/{len(paths)}] {path.name}", end="\r")
    print()

    scores_csv = out_dir / "check_d_content_sensitivity_scores.csv"
    with scores_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    d0 = np.array([r["D0"] for r in rows], dtype=np.float64)
    deltas = np.array([r["delta"] for r in rows], dtype=np.float64)
    luminances = np.array([r["luminance"] for r in rows], dtype=np.float64)
    edges = np.array([r["edge_density"] for r in rows], dtype=np.float64)
    rho_lum = spearmanr(d0, luminances)
    rho_edge = spearmanr(d0, edges)

    report_rows = [
        report_row("SD of D0 across images", float(d0.std(ddof=1))),
        report_row("mean of delta", float(deltas.mean())),
        report_row("median of delta", float(np.median(deltas))),
        report_row("fraction with delta > 0", float(np.mean(deltas > 0))),
        report_row("Spearman(D0, luminance)", float(rho_lum.statistic)),
        report_row("Spearman(D0, edge density)", float(rho_edge.statistic)),
    ]

    report_csv = out_dir / "check_d_report_table.csv"
    with report_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quantity", "value"])
        writer.writeheader()
        writer.writerows(report_rows)

    summary = {
        "check": "Task B - Content sensitivity test (Check D)",
        "data": args.data,
        "checkpoint": args.ckpt,
        "num_images": len(rows),
        "score_name": "D_WACV = S_Z; higher means worse quality",
        "image_size": int(img_size),
        "latent_dim": int(ldim),
        "sz_mode": cfg.get("sz_mode", "mu_only"),
        "sz_sigma_t_max": float(cfg.get("sz_sigma_t_max", 1.0)),
        "blur_level_4": BLUR_LEVEL_4,
        "content_proxy_note": "Computed on clean 256x256 center crop after the same resize/crop used for scoring.",
        "report_table": report_rows,
        "spearman_D0_luminance_pvalue": float(rho_lum.pvalue),
        "spearman_D0_edge_density_pvalue": float(rho_edge.pvalue),
        "scores_csv": str(scores_csv),
        "report_csv": str(report_csv),
        "summary_txt": str(out_dir / "check_d_summary.txt"),
        "clean_crops_dir": str(clean_dir),
        "blur_level4_dir": str(blur_dir),
    }

    summary_json = out_dir / "check_d_summary.json"
    summary_txt = out_dir / "check_d_summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_report_txt(summary_txt, report_rows)

    print(summary_txt.read_text())
    print(f"Wrote scores CSV: {scores_csv}")
    print(f"Wrote report CSV: {report_csv}")
    print(f"Wrote summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
