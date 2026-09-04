#!/usr/bin/env python3
"""Task E: metric correspondence on frozen DRealSR SR outputs."""

import argparse
import csv
import json
import math
import re
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
METRIC_COLUMNS = [
    "method",
    "image_id",
    "sr_path",
    "hr_path",
    "lr_path",
    "D_WACV",
    "NIQE",
    "LR_content",
    "PSNR",
    "SSIM",
    "LPIPS",
    "DISTS",
]


def collect_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def strip_hr_id(path):
    return re.sub(r"_x4$", "", Path(path).stem)


def strip_swin_id(path):
    stem = Path(path).stem
    prefix = "DRealSR_all_x4_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return re.sub(r"_x1$", "", stem)


def strip_hat_id(path):
    stem = Path(path).stem
    return re.sub(r"_x1_DReal_x4$", "", stem)


def lr_number(path):
    match = re.search(r"dreal4_(\d+)$", Path(path).stem)
    if not match:
        return None
    return int(match.group(1))


def build_hr_map(hr_dir):
    return {strip_hr_id(p): p for p in collect_images(hr_dir)}


def build_sr_map(sr_dir, method):
    if method == "SwinSR":
        return {strip_swin_id(p): p for p in collect_images(sr_dir)}
    if method == "HAT":
        return {strip_hat_id(p): p for p in collect_images(sr_dir)}
    raise ValueError(f"Unknown method: {method}")


def build_lr_map(lr_dir, hr_map):
    lr_paths = collect_images(lr_dir)
    named = {
        re.sub(r"_x1$", "", p.stem): p
        for p in lr_paths
        if not p.stem.startswith("dreal4_")
    }
    if named:
        return named

    numbered = [p for p in lr_paths if lr_number(p) is not None]
    numbered = sorted(numbered, key=lr_number)
    hr_ids = [strip_hr_id(p) for p in sorted(hr_map.values(), key=lambda x: x.name)]
    if len(numbered) != len(hr_ids):
        raise SystemExit(
            f"Cannot map numbered LR files: got {len(numbered)} LR files and {len(hr_ids)} HR files."
        )
    return dict(zip(hr_ids, numbered))


def pil_rgb(path):
    return Image.open(path).convert("RGB")


def pil_to_tensor(image, device):
    arr = (np.asarray(image).astype(np.float32) / 255.0).clip(0.0, 1.0)
    return transforms.ToTensor()(arr).unsqueeze(0).to(device)


def center_metric_crop(image, crop_size):
    if crop_size <= 0:
        return image
    if image.width < crop_size or image.height < crop_size:
        image = _resize_short_side(image, crop_size)
    return center_crop(image, crop_size)


def score_wacv(generator, image, mu_ref, Sigma_ref, img_size, cfg, device):
    crop = center_crop(_resize_short_side(image, img_size), img_size)
    x = pil_to_tensor(crop, device)
    score = score_sz_eval(
        generator,
        x,
        mu_ref,
        Sigma_ref,
        sigma_t_max=cfg.get("sz_sigma_t_max", 1.0),
        mu_only=(cfg.get("sz_mode", "mu_only") == "mu_only"),
    )
    return float(score.view(-1)[0].item())


def psnr_rgb(sr_crop, hr_crop):
    sr = np.asarray(sr_crop).astype(np.float32) / 255.0
    hr = np.asarray(hr_crop).astype(np.float32) / 255.0
    mse = float(np.mean((sr - hr) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def ssim_rgb(sr_crop, hr_crop):
    sr = np.asarray(sr_crop).astype(np.float32) / 255.0
    hr = np.asarray(hr_crop).astype(np.float32) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    vals = []
    for ch in range(3):
        x = sr[:, :, ch]
        y = hr[:, :, ch]
        mux = cv2.GaussianBlur(x, (11, 11), 1.5)
        muy = cv2.GaussianBlur(y, (11, 11), 1.5)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigx2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mux2
        sigy2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - muy2
        sigxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - muxy
        ssim_map = ((2 * muxy + c1) * (2 * sigxy + c2)) / (
            (mux2 + muy2 + c1) * (sigx2 + sigy2 + c2)
        )
        vals.append(float(np.mean(ssim_map)))
    return float(np.mean(vals))


def lr_content(sr_image, lr_image):
    down = sr_image.resize((sr_image.width // 4, sr_image.height // 4))
    if down.size != lr_image.size:
        raise ValueError(f"Down4(SR) size {down.size} does not match LR size {lr_image.size}")
    down_arr = np.asarray(down).astype(np.float32) / 255.0
    lr_arr = np.asarray(lr_image).astype(np.float32) / 255.0
    return float(np.sqrt(np.mean((down_arr - lr_arr) ** 2)))


def metric_tensor(image, crop_size, device):
    crop = center_metric_crop(image, crop_size)
    return pil_to_tensor(crop, device)


def load_existing(csv_path):
    if not csv_path.is_file():
        return {}, []
    rows = []
    done = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            done[row["image_id"]] = row
    return done, rows


def write_rows(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def score_method(
    method,
    sr_map,
    hr_map,
    lr_map,
    generator,
    mu_ref,
    Sigma_ref,
    img_size,
    cfg,
    device,
    pyiqa_metrics,
    metric_crop,
    out_csv,
):
    existing, rows = load_existing(out_csv)
    image_ids = sorted(set(sr_map) & set(hr_map) & set(lr_map))
    missing = {
        "sr_missing": sorted((set(hr_map) & set(lr_map)) - set(sr_map)),
        "hr_missing": sorted((set(sr_map) & set(lr_map)) - set(hr_map)),
        "lr_missing": sorted((set(sr_map) & set(hr_map)) - set(lr_map)),
    }

    by_id = {r["image_id"]: r for r in rows}
    for idx, image_id in enumerate(image_ids, start=1):
        if image_id in existing:
            print(f"[{method} {idx}/{len(image_ids)}] {image_id} cached", end="\r")
            continue

        sr_image = pil_rgb(sr_map[image_id])
        hr_image = pil_rgb(hr_map[image_id])
        lr_image = pil_rgb(lr_map[image_id])
        sr_crop = center_metric_crop(sr_image, metric_crop)
        hr_crop = center_metric_crop(hr_image, metric_crop)
        sr_t = pil_to_tensor(sr_crop, device)
        hr_t = pil_to_tensor(hr_crop, device)

        with torch.no_grad():
            row = {
                "method": method,
                "image_id": image_id,
                "sr_path": str(sr_map[image_id]),
                "hr_path": str(hr_map[image_id]),
                "lr_path": str(lr_map[image_id]),
                "D_WACV": score_wacv(generator, sr_image, mu_ref, Sigma_ref, img_size, cfg, device),
                "NIQE": float(pyiqa_metrics["niqe"](sr_t).view(-1)[0].item()),
                "LR_content": lr_content(sr_image, lr_image),
                "PSNR": psnr_rgb(sr_crop, hr_crop),
                "SSIM": ssim_rgb(sr_crop, hr_crop),
                "LPIPS": float(pyiqa_metrics["lpips"](sr_t, hr_t).view(-1)[0].item()),
                "DISTS": float(pyiqa_metrics["dists"](sr_t, hr_t).view(-1)[0].item()),
            }

        by_id[image_id] = row
        rows = [by_id[i] for i in sorted(by_id)]
        write_rows(out_csv, rows)
        print(f"[{method} {idx}/{len(image_ids)}] {image_id}", end="\r")

    print()
    rows = [by_id[i] for i in sorted(by_id)]
    write_rows(out_csv, rows)
    return rows, image_ids, missing


def finite_pair(rows, metric):
    x, y = [], []
    for row in rows:
        try:
            xv = float(row["D_WACV"])
            yv = float(row[metric])
        except (TypeError, ValueError):
            continue
        if math.isfinite(xv) and math.isfinite(yv):
            x.append(xv)
            y.append(yv)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def correlation_rows(rows):
    out = []
    for metric in ["NIQE", "LR_content", "PSNR", "SSIM", "LPIPS", "DISTS"]:
        x, y = finite_pair(rows, metric)
        if len(x) >= 3:
            corr = spearmanr(x, y)
            rho = float(corr.statistic)
            pvalue = float(corr.pvalue)
        else:
            rho = float("nan")
            pvalue = float("nan")
        out.append(
            {
                "pair": f"D_WACV vs {metric}",
                "Spearman_rho": rho,
                "N": int(len(x)),
                "p_value": pvalue,
            }
        )
    return out


def write_summary_txt(path, corr_rows, summary):
    lines = [
        "Task E - Metric correspondence on frozen SR corpus",
        "Scoring only; no training.",
        f"Frozen WACV checkpoint: {summary['checkpoint']}",
        f"Metric crop size: {summary['metric_crop_size']}",
        f"pyiqa version: {summary['pyiqa_version']}",
        "",
        "pair,Spearman_rho,N",
    ]
    for row in corr_rows:
        lines.append(f"{row['pair']},{row['Spearman_rho']:.6f},{row['N']}")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--swin_dir",
        default="/data/projectwork/sr_output/swinIr_Inference/SwinIR_results/Dreal",
    )
    parser.add_argument(
        "--hat_dir",
        default="/data/projectwork/sr_output/HAT_inference/DReal_x4/visualization/DReal_x4",
    )
    parser.add_argument(
        "--hr_dir",
        default="/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/DRealSR-20231109T095957Z-004/DRealSR/Test_x4/HR",
    )
    parser.add_argument(
        "--lr_dir",
        default="/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/DRealSR-20231109T095957Z-004/DRealSR/Test_x4/LR",
    )
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"),
    )
    parser.add_argument("--out_dir", default=str(ROOT / "day2_tasks" / "task_e_metric_correspondence"))
    parser.add_argument("--metric_crop", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    try:
        import pyiqa
    except ImportError as exc:
        raise SystemExit(
            "pyiqa is required for NIQE/LPIPS/DISTS. Run with "
            "/home/projectwork/.conda/envs/vae/bin/python."
        ) from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")

    generator, mu_ref, Sigma_ref, img_size, ldim, cfg = load_best(args.ckpt, device)
    generator.eval()

    hr_map = build_hr_map(args.hr_dir)
    lr_map = build_lr_map(args.lr_dir, hr_map)
    method_maps = {
        "SwinSR": build_sr_map(args.swin_dir, "SwinSR"),
        "HAT": build_sr_map(args.hat_dir, "HAT"),
    }

    pyiqa_metrics = {
        "niqe": pyiqa.create_metric("niqe", device=device),
        "lpips": pyiqa.create_metric("lpips", device=device),
        "dists": pyiqa.create_metric("dists", device=device),
    }

    all_rows = []
    match_info = {}
    for method, sr_map in method_maps.items():
        rows, matched_ids, missing = score_method(
            method,
            sr_map,
            hr_map,
            lr_map,
            generator,
            mu_ref,
            Sigma_ref,
            img_size,
            cfg,
            device,
            pyiqa_metrics,
            args.metric_crop,
            out_dir / f"task_e_{method.lower()}_metrics.csv",
        )
        all_rows.extend(rows)
        match_info[method] = {
            "num_sr": len(sr_map),
            "num_matched": len(matched_ids),
            "missing": missing,
        }

    pooled_csv = out_dir / "task_e_pooled_metrics.csv"
    write_rows(pooled_csv, all_rows)

    corr = correlation_rows(all_rows)
    corr_csv = out_dir / "task_e_pooled_correlations.csv"
    with corr_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "Spearman_rho", "N", "p_value"])
        writer.writeheader()
        writer.writerows(corr)

    summary = {
        "check": "Task E - Metric correspondence on frozen SR corpus",
        "scoring_only_no_training": True,
        "checkpoint": args.ckpt,
        "score_name": "D_WACV = S_Z; higher means worse quality",
        "wacv_preprocess": f"resize short side only if < {img_size}, then center crop {img_size}x{img_size}",
        "metric_crop_size": args.metric_crop,
        "metric_crop_note": "NIQE, PSNR, SSIM, LPIPS, and DISTS are computed on the same center crop size for speed/reproducibility.",
        "lr_content_definition": "RGB RMSE in [0,1] between PIL Down4(SR) and matched LR image",
        "lr_mapping_note": "LR files are numbered dreal4_1..93; mapped to HR IDs by lexicographic HR order from the original Test_x4 zip/listing.",
        "pyiqa_version": getattr(pyiqa, "__version__", "unknown"),
        "image_size": int(img_size),
        "latent_dim": int(ldim),
        "sz_mode": cfg.get("sz_mode", "mu_only"),
        "sz_sigma_t_max": float(cfg.get("sz_sigma_t_max", 1.0)),
        "inputs": {
            "SwinSR": args.swin_dir,
            "HAT": args.hat_dir,
            "HR": args.hr_dir,
            "LR": args.lr_dir,
        },
        "matches": match_info,
        "correlations": corr,
        "outputs": {
            "SwinSR_csv": str(out_dir / "task_e_swinsr_metrics.csv"),
            "HAT_csv": str(out_dir / "task_e_hat_metrics.csv"),
            "pooled_csv": str(pooled_csv),
            "correlations_csv": str(corr_csv),
            "summary_txt": str(out_dir / "task_e_summary.txt"),
            "summary_json": str(out_dir / "task_e_summary.json"),
        },
    }

    summary_json = out_dir / "task_e_summary.json"
    summary_txt = out_dir / "task_e_summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_summary_txt(summary_txt, corr, summary)

    print(summary_txt.read_text())
    print(f"Wrote: {pooled_csv}")
    print(f"Wrote: {corr_csv}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()
