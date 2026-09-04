#!/usr/bin/env python3
"""Task 4: score KADID-10k with frozen WACV and IQA metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torchvision import transforms


ROOT = Path("/home/projectwork/student_package")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))

from dataloader import _resize_short_side, center_crop  # noqa: E402
from infer import load_best  # noqa: E402
from score import score_sz_eval  # noqa: E402


OUT_DIR = ROOT / "day2_tasks" / "kaddik"
DEFAULT_MANIFEST = OUT_DIR / "kadid10k_manifest.csv"
DEFAULT_CKPT = ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"
SCORE_COLUMNS = [
    "dataset",
    "ref_id",
    "distorted_path",
    "ref_path",
    "distortion_type",
    "severity_or_level",
    "mos_or_dmos",
    "D_WACV",
    "PSNR",
    "SSIM",
    "NIQE",
    "LPIPS",
]


def pil_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    arr = (np.asarray(image).astype(np.float32) / 255.0).clip(0.0, 1.0)
    return transforms.ToTensor()(arr).unsqueeze(0).to(device)


def center_metric_crop(image: Image.Image, crop_size: int) -> Image.Image:
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


def psnr_rgb(dist_crop: Image.Image, ref_crop: Image.Image) -> float:
    dist = np.asarray(dist_crop).astype(np.float32) / 255.0
    ref = np.asarray(ref_crop).astype(np.float32) / 255.0
    mse = float(np.mean((dist - ref) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def ssim_rgb(dist_crop: Image.Image, ref_crop: Image.Image) -> float:
    dist = np.asarray(dist_crop).astype(np.float32) / 255.0
    ref = np.asarray(ref_crop).astype(np.float32) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    vals = []
    for ch in range(3):
        x = dist[:, :, ch]
        y = ref[:, :, ch]
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


def load_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    done = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done[row["distorted_path"]] = row
    return done


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    tmp = path.with_suffix(".tmp.csv")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def finite_pairs(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(df[metric], errors="coerce")
    y = pd.to_numeric(df["mos_or_dmos"], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask].to_numpy(dtype=np.float64), y[mask].to_numpy(dtype=np.float64)


def metric_correlation_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for metric in ["D_WACV", "PSNR", "SSIM", "NIQE", "LPIPS"]:
        x, y = finite_pairs(df, metric)
        if len(x) >= 3:
            srcc = spearmanr(x, y)
            plcc = pearsonr(x, y)
            rows.append(
                {
                    "metric": metric,
                    "target": "mos_or_dmos",
                    "N": int(len(x)),
                    "spearman_rho": float(srcc.statistic),
                    "spearman_pvalue": float(srcc.pvalue),
                    "pearson_r": float(plcc.statistic),
                    "pearson_pvalue": float(plcc.pvalue),
                }
            )
        else:
            rows.append(
                {
                    "metric": metric,
                    "target": "mos_or_dmos",
                    "N": int(len(x)),
                    "spearman_rho": float("nan"),
                    "spearman_pvalue": float("nan"),
                    "pearson_r": float("nan"),
                    "pearson_pvalue": float("nan"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out_csv", default=str(OUT_DIR / "kadid10k_task4_scores.csv"))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--metric_crop", type=int, default=256)
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: score first N manifest rows.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    import pyiqa

    manifest = pd.read_csv(args.manifest)
    if args.limit > 0:
        manifest = manifest.head(args.limit)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

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
    }

    existing = load_existing(out_csv)
    rows_by_path: dict[str, dict[str, object]] = {k: dict(v) for k, v in existing.items()}
    ref_cache: dict[str, Image.Image] = {}
    processed_since_flush = 0

    with torch.no_grad():
        for idx, rec in enumerate(manifest.to_dict("records"), start=1):
            distorted_path = rec["distorted_path"]
            if distorted_path in rows_by_path:
                print(f"[{idx}/{len(manifest)}] cached", end="\r")
                continue

            dist_img = pil_rgb(distorted_path)
            ref_path = rec["ref_path"]
            if ref_path not in ref_cache:
                ref_cache[ref_path] = pil_rgb(ref_path)
            ref_img = ref_cache[ref_path]

            dist_crop = center_metric_crop(dist_img, args.metric_crop)
            ref_crop = center_metric_crop(ref_img, args.metric_crop)
            dist_t = pil_to_tensor(dist_crop, device)
            ref_t = pil_to_tensor(ref_crop, device)

            row = {
                "dataset": rec["dataset"],
                "ref_id": rec["ref_id"],
                "distorted_path": distorted_path,
                "ref_path": ref_path,
                "distortion_type": rec["distortion_type"],
                "severity_or_level": int(rec["severity_or_level"]),
                "mos_or_dmos": float(rec["mos_or_dmos"]),
                "D_WACV": score_wacv(generator, dist_img, mu_ref, sigma_ref, img_size, cfg, device),
                "PSNR": psnr_rgb(dist_crop, ref_crop),
                "SSIM": ssim_rgb(dist_crop, ref_crop),
                "NIQE": float(pyiqa_metrics["niqe"](dist_t).view(-1)[0].item()),
                "LPIPS": float(pyiqa_metrics["lpips"](dist_t, ref_t).view(-1)[0].item()),
            }
            rows_by_path[distorted_path] = row
            processed_since_flush += 1
            print(f"[{idx}/{len(manifest)}] {Path(distorted_path).name}", end="\r")

            if processed_since_flush >= args.flush_every:
                ordered = [rows_by_path[p] for p in manifest["distorted_path"] if p in rows_by_path]
                write_rows(out_csv, ordered)
                processed_since_flush = 0

    ordered = [rows_by_path[p] for p in manifest["distorted_path"] if p in rows_by_path]
    write_rows(out_csv, ordered)
    print()

    scored = pd.read_csv(out_csv)
    corr_rows = metric_correlation_rows(scored)
    corr_csv = OUT_DIR / "kadid10k_task4_metric_correlations.csv"
    pd.DataFrame(corr_rows).to_csv(corr_csv, index=False)

    summary = {
        "task": "Task 4: KADID-10k scoring",
        "manifest": str(args.manifest),
        "output_csv": str(out_csv),
        "correlations_csv": str(corr_csv),
        "num_manifest_rows": int(len(manifest)),
        "num_scored_rows": int(len(scored)),
        "checkpoint": args.ckpt,
        "score_name": "D_WACV = S_Z; higher means worse quality",
        "wacv_preprocess": f"resize short side only if < {img_size}, then center crop {img_size}x{img_size}",
        "metric_crop_size": args.metric_crop,
        "metric_note": "PSNR, SSIM, NIQE, and LPIPS were computed on the same center 256x256 crop for consistency with previous Task E scoring.",
        "pyiqa_version": getattr(pyiqa, "__version__", "unknown"),
        "image_size": int(img_size),
        "latent_dim": int(ldim),
        "sz_mode": cfg.get("sz_mode", "mu_only"),
        "sz_sigma_t_max": float(cfg.get("sz_sigma_t_max", 1.0)),
        "correlations_against_mos_or_dmos": corr_rows,
    }
    summary_json = OUT_DIR / "kadid10k_task4_summary.json"
    summary_txt = OUT_DIR / "kadid10k_task4_summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    summary_txt.write_text(
        "\n".join(
            [
                "Task 4: KADID-10k scoring",
                f"Rows scored: {len(scored)} / {len(manifest)}",
                f"Scores CSV: {out_csv}",
                f"Correlations CSV: {corr_csv}",
                f"Frozen WACV checkpoint: {args.ckpt}",
                f"WACV preprocess: {summary['wacv_preprocess']}",
                f"Metric crop size: {args.metric_crop}",
                "Metrics: D_WACV, PSNR, SSIM, NIQE, LPIPS, MOS/DMOS",
                "",
                "Correlation against mos_or_dmos:",
            ]
            + [
                f"{r['metric']}: Spearman={r['spearman_rho']:.6f}, Pearson={r['pearson_r']:.6f}, N={r['N']}"
                for r in corr_rows
            ]
        )
        + "\n"
    )
    print(summary_txt.read_text())


if __name__ == "__main__":
    main()
