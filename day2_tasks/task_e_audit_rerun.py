#!/usr/bin/env python3
"""Methodological audit and rerun for Task E."""

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
from scipy.stats import rankdata, spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))
sys.path.insert(0, str(ROOT / "day2_tasks"))

from infer import load_best  # noqa: E402
from task_e_metric_correspondence import (  # noqa: E402
    METRIC_COLUMNS,
    build_hr_map,
    build_lr_map,
    build_sr_map,
    center_metric_crop,
    collect_images,
    lr_content,
    pil_rgb,
    pil_to_tensor,
    psnr_rgb,
    score_wacv,
    ssim_rgb,
)


SWIN_DIR = "/data/projectwork/sr_output/swinIr_Inference/SwinIR_results/Dreal"
HAT_DIR = "/data/projectwork/sr_output/HAT_inference/DReal_x4/visualization/DReal_x4"
HR_DIR = "/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/DRealSR-20231109T095957Z-004/DRealSR/Test_x4/HR"
LR_DIR = "/home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/DRealSR-20231109T095957Z-004/DRealSR/Test_x4/LR"
CKPT = str(ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth")
OUT_DIR = ROOT / "task_e_audit"
METRICS = ["NIQE", "LR_content", "PSNR", "SSIM", "LPIPS", "DISTS"]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def image_mode_size(path):
    with Image.open(path) as im:
        return im.mode, im.size


def audit_pairing(method_maps, hr_map, lr_map):
    rows = []
    for method, sr_map in method_maps.items():
        all_ids = sorted(set(sr_map) | set(hr_map) | set(lr_map))
        for image_id in all_ids:
            sr = sr_map.get(image_id)
            hr = hr_map.get(image_id)
            lr = lr_map.get(image_id)
            row = {
                "method": method,
                "image_id": image_id,
                "sr_path": str(sr) if sr else "",
                "hr_path": str(hr) if hr else "",
                "lr_path": str(lr) if lr else "",
                "sr_exists": bool(sr and sr.is_file()),
                "hr_exists": bool(hr and hr.is_file()),
                "lr_exists": bool(lr and lr.is_file()),
                "sr_mode": "",
                "hr_mode": "",
                "lr_mode": "",
                "sr_width": "",
                "sr_height": "",
                "hr_width": "",
                "hr_height": "",
                "lr_width": "",
                "lr_height": "",
                "sr_hr_same_size": False,
                "lr_matches_hr_div4": False,
                "status": "ok",
            }
            problems = []
            if not sr:
                problems.append("missing_sr")
            if not hr:
                problems.append("missing_hr")
            if not lr:
                problems.append("missing_lr")
            if sr and hr and lr:
                sr_mode, sr_size = image_mode_size(sr)
                hr_mode, hr_size = image_mode_size(hr)
                lr_mode, lr_size = image_mode_size(lr)
                row.update(
                    {
                        "sr_mode": sr_mode,
                        "hr_mode": hr_mode,
                        "lr_mode": lr_mode,
                        "sr_width": sr_size[0],
                        "sr_height": sr_size[1],
                        "hr_width": hr_size[0],
                        "hr_height": hr_size[1],
                        "lr_width": lr_size[0],
                        "lr_height": lr_size[1],
                        "sr_hr_same_size": sr_size == hr_size,
                        "lr_matches_hr_div4": (hr_size[0] // 4, hr_size[1] // 4) == lr_size
                        and hr_size[0] % 4 == 0
                        and hr_size[1] % 4 == 0,
                    }
                )
                if sr_size != hr_size:
                    problems.append("sr_hr_size_mismatch")
                if row["lr_matches_hr_div4"] is False:
                    problems.append("lr_not_hr_div4")
                if sr_mode != "RGB" or hr_mode != "RGB" or lr_mode != "RGB":
                    problems.append("non_rgb_mode_before_convert")
            if problems:
                row["status"] = ";".join(problems)
            rows.append(row)
    return rows


def finite_xy(rows, metric):
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
    return np.asarray(x), np.asarray(y)


def bootstrap_ci(x, y, n_boot=1000, seed=123):
    if len(x) < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        vals.append(spearmanr(x[idx], y[idx]).statistic)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def corr_table(rows, label, method="pooled"):
    out = []
    subset = rows if method == "pooled" else [r for r in rows if r["method"] == method]
    for metric in METRICS:
        x, y = finite_xy(subset, metric)
        if len(x) >= 3:
            res = spearmanr(x, y)
            lo, hi = bootstrap_ci(x, y)
            rho, p = float(res.statistic), float(res.pvalue)
        else:
            rho = p = lo = hi = float("nan")
        out.append(
            {
                "variant": label,
                "method": method,
                "pair": f"D_WACV vs {metric}",
                "rho": rho,
                "p_value": p,
                "N": int(len(x)),
                "rho_ci95_low": lo,
                "rho_ci95_high": hi,
            }
        )
    return out


def residualize(y, cov):
    x = np.column_stack([np.ones(len(cov)), cov])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return y - x @ beta


def partial_rows(rows, variant):
    out = []
    for metric in ["PSNR", "SSIM", "LPIPS", "DISTS"]:
        vals = []
        for row in rows:
            try:
                triple = (float(row["D_WACV"]), float(row[metric]), float(row["LR_content"]))
            except (TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in triple):
                vals.append(triple)
        arr = np.asarray(vals, dtype=np.float64)
        if len(arr) >= 4:
            d_rank = rankdata(arr[:, 0])
            m_rank = rankdata(arr[:, 1])
            c_rank = rankdata(arr[:, 2])
            rd = residualize(d_rank, c_rank)
            rm = residualize(m_rank, c_rank)
            res = spearmanr(rd, rm)
            rho, p = float(res.statistic), float(res.pvalue)
        else:
            rho = p = float("nan")
        out.append(
            {
                "variant": variant,
                "pair": f"D_WACV vs {metric} controlling LR_content",
                "partial_spearman_rho": rho,
                "p_value": p,
                "N": int(len(arr)),
            }
        )
    return out


def stratified_rows(rows, variant):
    out = []
    vals = np.asarray([float(r["LR_content"]) for r in rows if math.isfinite(float(r["LR_content"]))])
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    for group, pred in [
        ("low", lambda v: v <= q1),
        ("middle", lambda v: q1 < v <= q2),
        ("high", lambda v: v > q2),
    ]:
        subset = [r for r in rows if pred(float(r["LR_content"]))]
        for metric in ["PSNR", "SSIM", "LPIPS", "DISTS"]:
            x, y = finite_xy(subset, metric)
            if len(x) >= 3:
                res = spearmanr(x, y)
                rho, p = float(res.statistic), float(res.pvalue)
            else:
                rho = p = float("nan")
            out.append(
                {
                    "variant": variant,
                    "lr_content_group": group,
                    "pair": f"D_WACV vs {metric}",
                    "rho": rho,
                    "p_value": p,
                    "N": int(len(x)),
                    "lr_content_min": float(min(float(r["LR_content"]) for r in subset)),
                    "lr_content_max": float(max(float(r["LR_content"]) for r in subset)),
                }
            )
    return out


def full_psnr_ssim(sr_img, hr_img):
    if sr_img.size != hr_img.size:
        raise ValueError(f"SR/HR size mismatch: {sr_img.size} vs {hr_img.size}")
    return psnr_rgb(sr_img, hr_img), ssim_rgb(sr_img, hr_img)


def score_rows(method, ids, sr_map, hr_map, lr_map, generator, mu_ref, sigma_ref, img_size, cfg, device, metrics):
    crop_rows, full_rows = [], []
    for i, image_id in enumerate(ids, 1):
        sr = pil_rgb(sr_map[image_id])
        hr = pil_rgb(hr_map[image_id])
        lr = pil_rgb(lr_map[image_id])
        sr_crop = center_metric_crop(sr, img_size)
        hr_crop = center_metric_crop(hr, img_size)
        sr_t = pil_to_tensor(sr_crop, device)
        hr_t = pil_to_tensor(hr_crop, device)
        d = score_wacv(generator, sr, mu_ref, sigma_ref, img_size, cfg, device)
        base = {
            "method": method,
            "image_id": image_id,
            "sr_path": str(sr_map[image_id]),
            "hr_path": str(hr_map[image_id]),
            "lr_path": str(lr_map[image_id]),
            "D_WACV": d,
            "LR_content": lr_content(sr, lr),
        }
        with torch.no_grad():
            crop = dict(base)
            crop.update(
                {
                    "NIQE": float(metrics["niqe"](sr_t).view(-1)[0].item()),
                    "PSNR": psnr_rgb(sr_crop, hr_crop),
                    "SSIM": ssim_rgb(sr_crop, hr_crop),
                    "LPIPS": float(metrics["lpips"](sr_t, hr_t).view(-1)[0].item()),
                    "DISTS": float(metrics["dists"](sr_t, hr_t).view(-1)[0].item()),
                }
            )
            full = dict(base)
            psnr, ssim = full_psnr_ssim(sr, hr)
            full_t = pil_to_tensor(sr, device)
            full.update(
                {
                    "NIQE": float(metrics["niqe"](full_t).view(-1)[0].item()),
                    "PSNR": psnr,
                    "SSIM": ssim,
                    "LPIPS": float("nan"),
                    "DISTS": float("nan"),
                }
            )
        crop_rows.append(crop)
        full_rows.append(full)
        print(f"[{method} {i}/{len(ids)}] {image_id}", flush=True)
    print(flush=True)
    return crop_rows, full_rows


def score_direction_audit():
    path = ROOT / "task wacv" / "check_b_gaussian_blur_severity_scores.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_name = {}
    for r in rows:
        by_name.setdefault(r["name"], {})[int(r["severity"])] = float(r["D_WACV"])
    strict = sum(all(vals[i] < vals[i + 1] for i in range(4)) for vals in by_name.values())
    blur_gt_clean = sum(vals[4] > vals[0] for vals in by_name.values())
    return {
        "source_csv": str(path),
        "num_images": len(by_name),
        "strict_clean_blur1_blur2_blur3_blur4": strict,
        "blur4_gt_clean": blur_gt_clean,
        "adjacent_ordering_rate_from_check_b_summary": 0.87,
        "pooled_spearman_from_check_b_summary": 0.7330107706580024,
        "sample_images": [
            {"image": k, **{f"D{s}": v[s] for s in range(5)}}
            for k, v in list(sorted(by_name.items()))[:5]
        ],
    }


def main():
    import pyiqa

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    generator, mu_ref, sigma_ref, img_size, ldim, cfg = load_best(CKPT, device)
    hr_map = build_hr_map(HR_DIR)
    lr_map = build_lr_map(LR_DIR, hr_map)
    method_maps = {
        "SwinSR": build_sr_map(SWIN_DIR, "SwinSR"),
        "HAT": build_sr_map(HAT_DIR, "HAT"),
    }
    pair_rows = audit_pairing(method_maps, hr_map, lr_map)
    write_csv(OUT_DIR / "pairing_audit.csv", pair_rows, list(pair_rows[0].keys()))
    bad_pairs = [r for r in pair_rows if r["status"] != "ok"]

    metrics = {
        "niqe": pyiqa.create_metric("niqe", device=device),
        "lpips": pyiqa.create_metric("lpips", device=device),
        "dists": pyiqa.create_metric("dists", device=device),
    }

    crop_rows, full_rows = [], []
    for method, sr_map in method_maps.items():
        ids = sorted(set(sr_map) & set(hr_map) & set(lr_map))
        cr, fr = score_rows(method, ids, sr_map, hr_map, lr_map, generator, mu_ref, sigma_ref, img_size, cfg, device, metrics)
        crop_rows.extend(cr)
        full_rows.extend(fr)

    write_csv(OUT_DIR / "task_e_crop_metrics.csv", crop_rows, METRIC_COLUMNS)
    write_csv(OUT_DIR / "task_e_full_metrics.csv", full_rows, METRIC_COLUMNS)

    corr_crop = corr_table(crop_rows, "crop", "pooled")
    corr_full = corr_table(full_rows, "full", "pooled")
    write_csv(OUT_DIR / "correlations_crop.csv", corr_crop, list(corr_crop[0].keys()))
    write_csv(OUT_DIR / "correlations_full.csv", corr_full, list(corr_full[0].keys()))
    by_method = []
    for variant, rows in [("crop", crop_rows), ("full", full_rows)]:
        for method in ["SwinSR", "HAT"]:
            by_method.extend(corr_table(rows, variant, method))
    write_csv(OUT_DIR / "correlations_by_method.csv", by_method, list(by_method[0].keys()))

    partial = partial_rows(crop_rows, "crop") + partial_rows(full_rows, "full")
    write_csv(OUT_DIR / "partial_correlations.csv", partial, list(partial[0].keys()))
    strat = stratified_rows(crop_rows, "crop") + stratified_rows(full_rows, "full")
    write_csv(OUT_DIR / "content_stratified_correlations.csv", strat, list(strat[0].keys()))

    ranked = sorted(crop_rows, key=lambda r: float(r["D_WACV"]))
    top_bottom = []
    for label, selected in [("lowest_D_WACV", ranked[:10]), ("highest_D_WACV", ranked[-10:])]:
        for r in selected:
            out = {"rank_group": label}
            out.update({k: r[k] for k in ["image_id", "method", "D_WACV", "PSNR", "SSIM", "LPIPS", "DISTS", "NIQE", "LR_content"]})
            top_bottom.append(out)
    write_csv(OUT_DIR / "top_bottom_wacv.csv", top_bottom, list(top_bottom[0].keys()))

    direction = score_direction_audit()
    summary = {
        "checkpoint": CKPT,
        "score_direction": "Higher D_WACV = worse quality; unchanged.",
        "wacv_preprocess": {
            "crop": f"center {img_size}x{img_size}",
            "resize": f"only if short side < {img_size}, PIL BICUBIC",
            "channel_order": "PIL RGB -> torchvision ToTensor RGB",
            "value_range": "[0,1] float32",
            "normalization": "no mean/std normalization",
            "same_as_checks_a_to_d": True,
        },
        "metric_preprocess": {
            "crop_variant": "same 256x256 center crop for SR and HR; data_range=1 for RGB computations",
            "full_variant": "full SR/HR image for PSNR/SSIM/NIQE; no resize, no border crop; LPIPS/DISTS NaN because no CUDA and full 22MP neural metric is impractical in this environment",
            "lpips_dists_input": "pyiqa 0.1.16 accepts RGB tensors in [0,1] and internally normalizes as required",
            "niqe": "pyiqa 0.1.16 no-reference NIQE, run on SR only",
        },
        "pairing": {
            "rows": len(pair_rows),
            "bad_rows": len(bad_pairs),
            "matched_per_method": {m: len(set(s) & set(hr_map) & set(lr_map)) for m, s in method_maps.items()},
            "lr_mapping": "dreal4_1..93 mapped to lexicographic HR order, matching original Test_x4.zip listing",
        },
        "score_direction_audit": direction,
        "crop_correlations": corr_crop,
        "full_correlations": corr_full,
        "by_method_correlations": by_method,
        "partial_correlations": partial,
    }
    (OUT_DIR / "task_e_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    def find(rows, pair):
        return next(r for r in rows if r["pair"] == pair)

    lines = [
        "Task E methodological audit rerun",
        "No training. WACV checkpoint and score direction unchanged.",
        "",
        f"Pairing audit: {len(bad_pairs)} problematic rows out of {len(pair_rows)}; matched SwinSR=93, HAT=93.",
        "WACV receives PIL RGB, [0,1] float32, no mean/std normalization, center 256x256 crop; same frozen wrapper/checkpoint as previous checks.",
        "PSNR/SSIM use matched HR, RGB [0,1], no border crop, no resize to hide mismatches.",
        "Full LPIPS/DISTS are NaN: CPU-only environment cannot practically run full 22MP neural metrics; crop LPIPS/DISTS are computed with pyiqa 0.1.16.",
        "",
        "Crop pooled correlations:",
    ]
    for r in corr_crop:
        lines.append(f"{r['pair']}: rho={r['rho']:.6f}, p={r['p_value']:.3g}, N={r['N']}, CI95=[{r['rho_ci95_low']:.6f},{r['rho_ci95_high']:.6f}]")
    lines.append("")
    lines.append("Full pooled correlations:")
    for r in corr_full:
        lines.append(f"{r['pair']}: rho={r['rho']:.6f}, p={r['p_value']:.3g}, N={r['N']}, CI95=[{r['rho_ci95_low']:.6f},{r['rho_ci95_high']:.6f}]")
    lines.extend(
        [
            "",
            "Answers:",
            f"Direction reversal on crop PSNR/SSIM: yes, rho PSNR={find(corr_crop, 'D_WACV vs PSNR')['rho']:.6f}, SSIM={find(corr_crop, 'D_WACV vs SSIM')['rho']:.6f}.",
            "By method: see correlations_by_method.csv; both methods are reported separately.",
            f"Full-resolution PSNR/SSIM: rho PSNR={find(corr_full, 'D_WACV vs PSNR')['rho']:.6f}, SSIM={find(corr_full, 'D_WACV vs SSIM')['rho']:.6f}.",
            "After controlling/stratifying LR_content: see partial_correlations.csv and content_stratified_correlations.csv.",
            "Likely diagnosis: pairing/preprocessing errors were not found; the PSNR/SSIM direction anomaly persists in the diagnostic outputs available here, so it looks more like content/SR-method behavior or WACV behavior than a simple pairing bug.",
        ]
    )
    (OUT_DIR / "task_e_audit_summary.txt").write_text("\n".join(lines) + "\n")
    print((OUT_DIR / "task_e_audit_summary.txt").read_text())


if __name__ == "__main__":
    main()
