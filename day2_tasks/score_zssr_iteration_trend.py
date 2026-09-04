#!/usr/bin/env python3
"""Score ZSSR intermediate iterations with frozen WACV."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

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


def collect_images(root):
    root = Path(root)
    rows = []
    for image_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for path in sorted(image_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
                continue
            match = re.search(r"iteration_(\d+)", path.stem)
            iteration = int(match.group(1)) if match else None
            rows.append(
                {
                    "folder": image_dir.name,
                    "iteration": iteration,
                    "filename": path.name,
                    "path": path,
                }
            )
    return sorted(rows, key=lambda r: (r["folder"], r["iteration"] if r["iteration"] is not None else -1, r["filename"]))


def pil_to_tensor(path, img_size, device):
    image = Image.open(path).convert("RGB")
    image = center_crop(_resize_short_side(image, img_size), img_size)
    arr = (np.asarray(image).astype(np.float32) / 255.0).clip(0.0, 1.0)
    return transforms.ToTensor()(arr).unsqueeze(0).to(device)


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "N": int(values.size),
        "mean": float(values.mean()),
        "SD": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/projectwork/vansh_zssr")
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"),
    )
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "day2_tasks" / "zssr itteration trend"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    per_folder_dir = out_dir / "per_folder_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_folder_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")

    generator, mu_ref, sigma_ref, img_size, ldim, cfg = load_best(args.ckpt, device)
    rows_in = collect_images(args.data)
    if not rows_in:
        raise SystemExit(f"No images found under {args.data}")

    rows = []
    with torch.no_grad():
        for idx, item in enumerate(rows_in, 1):
            x = pil_to_tensor(item["path"], img_size, device)
            d = score_sz_eval(
                generator,
                x,
                mu_ref,
                sigma_ref,
                sigma_t_max=cfg.get("sz_sigma_t_max", 1.0),
                mu_only=(cfg.get("sz_mode", "mu_only") == "mu_only"),
            )
            rows.append(
                {
                    "folder": item["folder"],
                    "iteration": item["iteration"],
                    "filename": item["filename"],
                    "path": str(item["path"]),
                    "D_WACV": float(d.view(-1)[0].item()),
                }
            )
            print(f"[{idx}/{len(rows_in)}] {item['folder']} {item['filename']}", end="\r")
    print()

    fieldnames = ["folder", "iteration", "filename", "path", "D_WACV"]
    all_csv = out_dir / "zssr_all_wacv_scores.csv"
    write_csv(all_csv, rows, fieldnames)

    folders = sorted({r["folder"] for r in rows})
    folder_summary = []
    for folder in folders:
        fr = [r for r in rows if r["folder"] == folder]
        write_csv(per_folder_dir / f"{safe_name(folder)}_wacv_scores.csv", fr, fieldnames)
        vals = [r["D_WACV"] for r in fr]
        stats = summarize_values(vals)
        iters = np.asarray([r["iteration"] for r in fr], dtype=np.float64)
        scores = np.asarray(vals, dtype=np.float64)
        rho = spearmanr(iters, scores) if len(fr) >= 3 else None
        first = fr[0]["D_WACV"]
        last = fr[-1]["D_WACV"]
        folder_summary.append(
            {
                "folder": folder,
                **stats,
                "first_iteration": fr[0]["iteration"],
                "first_D_WACV": first,
                "last_iteration": fr[-1]["iteration"],
                "last_D_WACV": last,
                "last_minus_first": last - first,
                "spearman_iteration_D_WACV": float(rho.statistic) if rho is not None else float("nan"),
                "spearman_p_value": float(rho.pvalue) if rho is not None else float("nan"),
            }
        )

    folder_summary_csv = out_dir / "zssr_folder_summary.csv"
    write_csv(folder_summary_csv, folder_summary, list(folder_summary[0].keys()))

    iteration_summary = []
    for iteration in sorted({r["iteration"] for r in rows}):
        ir = [r for r in rows if r["iteration"] == iteration]
        iteration_summary.append({"iteration": iteration, **summarize_values([r["D_WACV"] for r in ir])})
    iteration_summary_csv = out_dir / "zssr_iteration_summary.csv"
    write_csv(iteration_summary_csv, iteration_summary, list(iteration_summary[0].keys()))

    all_iters = np.asarray([r["iteration"] for r in rows], dtype=np.float64)
    all_scores = np.asarray([r["D_WACV"] for r in rows], dtype=np.float64)
    pooled_rho = spearmanr(all_iters, all_scores)
    summary = {
        "task": "ZSSR iteration trend with frozen WACV",
        "data": args.data,
        "checkpoint": args.ckpt,
        "score_name": "D_WACV = S_Z; higher means worse quality",
        "preprocess": f"PIL RGB, [0,1], resize short side only if < {img_size}, center crop {img_size}x{img_size}",
        "num_folders": len(folders),
        "num_images": len(rows),
        "iterations": sorted({r["iteration"] for r in rows}),
        "pooled_spearman_iteration_D_WACV": float(pooled_rho.statistic),
        "pooled_spearman_p_value": float(pooled_rho.pvalue),
        "outputs": {
            "all_scores_csv": str(all_csv),
            "folder_summary_csv": str(folder_summary_csv),
            "iteration_summary_csv": str(iteration_summary_csv),
            "per_folder_dir": str(per_folder_dir),
        },
    }
    (out_dir / "zssr_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "ZSSR iteration trend with frozen WACV",
        f"Data: {args.data}",
        f"Checkpoint: {args.ckpt}",
        "Score: D_WACV = S_Z; higher means worse quality",
        f"Folders: {len(folders)}",
        f"Images scored: {len(rows)}",
        f"Iterations: {summary['iterations']}",
        f"Pooled Spearman(iteration, D_WACV): {summary['pooled_spearman_iteration_D_WACV']:.6f}",
        f"Pooled p-value: {summary['pooled_spearman_p_value']:.6g}",
        "",
        "Iteration summary:",
        "iteration,N,mean,SD,median,min,max",
    ]
    for r in iteration_summary:
        lines.append(
            "{iteration},{N},{mean:.6f},{SD:.6f},{median:.6f},{min:.6f},{max:.6f}".format(**r)
        )
    lines.extend(
        [
            "",
            f"All scores CSV: {all_csv}",
            f"Folder summary CSV: {folder_summary_csv}",
            f"Iteration summary CSV: {iteration_summary_csv}",
            f"Per-folder CSVs: {per_folder_dir}",
        ]
    )
    (out_dir / "zssr_summary.txt").write_text("\n".join(lines) + "\n")
    print((out_dir / "zssr_summary.txt").read_text())


if __name__ == "__main__":
    main()
