#!/usr/bin/env python3
"""Check C: ballpark pristine HR-test scores against HR-calibration."""

import argparse
import csv
import json
import sys
from pathlib import Path

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
PATH_COLUMNS = ("path", "image_path", "filepath", "file_path", "filename", "file", "name", "image_id")


def collect_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def read_manifest(csv_path, root_dir):
    csv_path = Path(csv_path)
    root_dir = Path(root_dir)
    with csv_path.open(newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                value = None
                for col in PATH_COLUMNS:
                    if col in row and row[col]:
                        value = row[col].strip()
                        break
                if value:
                    rows.append(value)
        else:
            reader = csv.reader(f)
            rows = [row[0].strip() for row in reader if row and row[0].strip()]

    paths = []
    for value in rows:
        path = Path(value)
        if not path.is_absolute():
            path = root_dir / path
        paths.append(path)
    return paths


def paths_from_manifest_or_folder(csv_path, folder):
    if csv_path and Path(csv_path).is_file():
        paths = read_manifest(csv_path, folder)
    else:
        paths = collect_images(folder)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise SystemExit(f"Missing image files listed for {folder}:\n{preview}")
    if not paths:
        raise SystemExit(f"No images found for {folder}")
    return sorted(paths)


def pil_to_tensor(path, img_size, device):
    image = Image.open(path).convert("RGB")
    image = center_crop(_resize_short_side(image, img_size), img_size)
    arr = (np.asarray(image) / 255.0).astype("float32")
    return transforms.ToTensor()(arr).unsqueeze(0).to(device)


def score_split(split, paths, generator, mu_ref, Sigma_ref, img_size, cfg, device):
    rows = []
    with torch.no_grad():
        for idx, path in enumerate(paths, start=1):
            x = pil_to_tensor(path, img_size, device)
            score = score_sz_eval(
                generator,
                x,
                mu_ref,
                Sigma_ref,
                sigma_t_max=cfg.get("sz_sigma_t_max", 1.0),
                mu_only=(cfg.get("sz_mode", "mu_only") == "mu_only"),
            )
            rows.append(
                {
                    "split": split,
                    "image_id": path.name,
                    "D_WACV": float(score.view(-1)[0].item()),
                }
            )
            print(f"[{split} {idx}/{len(paths)}] {path.name}", end="\r")
    print()
    return rows


def summarize(split, values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "split": split,
        "N": int(values.size),
        "mean": float(values.mean()),
        "SD": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def write_summary_txt(path, summary_rows, delta_mean, threshold, passes):
    lines = [
        "VTask A: Ballpark pristine test (Check C)",
        "Score: D_WACV = S_Z; higher means worse quality",
        "",
        "summary table",
        "split,N,mean,SD,median,p05,p95",
    ]
    for row in summary_rows:
        lines.append(
            "{split},{N},{mean:.6f},{SD:.6f},{median:.6f},{p05:.6f},{p95:.6f}".format(**row)
        )
    lines.extend(
        [
            "",
            f"delta_mean = abs(mean_test - mean_cal) = {delta_mean:.6f}",
            f"threshold = 0.5 * SD_cal = {threshold:.6f}",
            f"passes_ballpark_rule = {passes}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calib_dir",
        default="/data/projectwork/swati_mam/HR_DATA/HR-calibration/DIV2K",
        help="Folder containing HR calibration DIV2K images.",
    )
    parser.add_argument(
        "--test_dir",
        default="/data/projectwork/swati_mam/HR_DATA/HR-test/DIV2K",
        help="Folder containing HR test DIV2K images.",
    )
    parser.add_argument("--calib_csv", default="hr_calibration.csv")
    parser.add_argument("--test_csv", default="hr_test.csv")
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
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")

    generator, mu_ref, Sigma_ref, img_size, ldim, cfg = load_best(args.ckpt, device)
    calib_paths = paths_from_manifest_or_folder(args.calib_csv, args.calib_dir)
    test_paths = paths_from_manifest_or_folder(args.test_csv, args.test_dir)

    rows = []
    rows.extend(score_split("calibration", calib_paths, generator, mu_ref, Sigma_ref, img_size, cfg, device))
    rows.extend(score_split("hr_test", test_paths, generator, mu_ref, Sigma_ref, img_size, cfg, device))

    scores_csv = out_dir / "check_c_ballpark_pristine_scores.csv"
    with scores_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_id", "D_WACV"])
        writer.writeheader()
        writer.writerows(rows)

    cal_scores = [r["D_WACV"] for r in rows if r["split"] == "calibration"]
    test_scores = [r["D_WACV"] for r in rows if r["split"] == "hr_test"]
    summary_rows = [
        summarize("calibration", cal_scores),
        summarize("hr_test", test_scores),
    ]
    delta_mean = abs(summary_rows[1]["mean"] - summary_rows[0]["mean"])
    threshold = 0.5 * summary_rows[0]["SD"]
    passes = bool(delta_mean <= threshold)

    summary_csv = out_dir / "check_c_summary_table.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "N", "mean", "SD", "median", "p05", "p95"])
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "check": "VTask A - Ballpark pristine test (Check C)",
        "checkpoint": args.ckpt,
        "calibration_dir": args.calib_dir,
        "test_dir": args.test_dir,
        "calibration_csv_used": str(args.calib_csv) if Path(args.calib_csv).is_file() else None,
        "test_csv_used": str(args.test_csv) if Path(args.test_csv).is_file() else None,
        "score_name": "D_WACV = S_Z; higher means worse quality",
        "image_size": int(img_size),
        "latent_dim": int(ldim),
        "sz_mode": cfg.get("sz_mode", "mu_only"),
        "sz_sigma_t_max": float(cfg.get("sz_sigma_t_max", 1.0)),
        "summary_table": summary_rows,
        "delta_mean": float(delta_mean),
        "threshold": float(threshold),
        "rule": "delta_mean <= 0.5 * SD_cal",
        "passes_ballpark_rule": passes,
        "scores_csv": str(scores_csv),
        "summary_csv": str(summary_csv),
    }

    summary_json = out_dir / "check_c_summary.json"
    summary_txt = out_dir / "check_c_summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    write_summary_txt(summary_txt, summary_rows, delta_mean, threshold, passes)

    print(summary_txt.read_text())
    print(f"Wrote scores CSV: {scores_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
