# Inference for CVAE IQA (S_Z score).
# Loads checkpoint format: best.pth with generator + mu_ref/Sigma_ref.
# Higher S_Z = worse quality.

import argparse
import csv
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr
from torchvision import transforms

warnings.filterwarnings("ignore", category=UserWarning)

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "external"))
from dataloader import _resize_short_side, center_crop  # noqa: E402
from model import CVAEGenerator_v2  # noqa: E402

from score import load_scoring_config, score_rem, score_sz_eval  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def save_intermediateResults(image1, image2, imageSavePath):
    width1, height1 = image1.size
    total_width = width1 + image2.size[0]
    merged_image = Image.new("RGB", (total_width, height1))
    merged_image.paste(image1, (0, 0))
    merged_image.paste(image2, (width1, 0))
    merged_image.save(imageSavePath)


def get_args():
    frozen = load_scoring_config()
    parser = argparse.ArgumentParser(description="Test / infer CVAE IQA (S_Z)")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="force CPU even if a GPU is available",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=frozen.get("checkpoint") or "checkpoints/best.pth",
        help="path to best.pth (self-contained with mu_ref/Sigma_ref)",
    )
    parser.add_argument("--image", type=str, default=None, help="single image path")
    parser.add_argument("--folder", type=str, default=None, help="folder of images")
    parser.add_argument("--img", type=int, default=None, help="override crop size")
    parser.add_argument("--ldim", type=int, default=None, help="override latent dim")
    parser.add_argument(
        "--metrics",
        type=str,
        default="sz",
        help="comma list: sz,rem  (S_Z higher=worse quality)",
    )
    parser.add_argument("--save_recon", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str, default="test_results")
    parser.add_argument(
        "--mos_csv",
        type=str,
        default=None,
        help="CSV with image filename + MOS for SROCC/PLCC/KROCC",
    )
    parser.add_argument("--mos_col", type=int, default=1, help="0-based MOS column")
    parser.add_argument("--name_col", type=int, default=0, help="0-based filename column")
    parser.add_argument("--csv_has_header", action="store_true", default=True)
    parser.add_argument("--no_csv_header", action="store_true", default=False)
    parser.add_argument(
        "--scoring_yaml",
        type=str,
        default="configs/scoring_frozen.yaml",
        help="frozen scoring rule (defaults); checkpoint sz_mode still wins if present",
    )
    opt = parser.parse_args()
    if opt.no_csv_header:
        opt.csv_has_header = False
    return opt


def pick_device(opt):
    if opt.cpu or not torch.cuda.is_available():
        if not opt.cpu and not torch.cuda.is_available():
            print("[WARN] CUDA not found; using CPU (slower)")
        return torch.device("cpu")
    torch.cuda.set_device("cuda:" + str(opt.gpu))
    device = torch.device("cuda:" + str(opt.gpu))
    print("CUDA Version:", torch.version.cuda)
    print(
        "current CUDA Device:",
        torch.cuda.current_device(),
        torch.cuda.get_device_name(torch.cuda.current_device()),
    )
    return device


def load_best(ckpt_path, device, img_override=None, ldim_override=None, scoring_yaml=None):
    frozen = load_scoring_config(scoring_yaml)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "generator" not in ckpt:
        raise KeyError(
            "Checkpoint missing 'generator' key. This script expects the NEW format "
            "(not old ['state_dict'])."
        )
    if "mu_ref" not in ckpt or "Sigma_ref" not in ckpt:
        raise KeyError(
            "best.pth must contain mu_ref and Sigma_ref for S_Z. "
            "Re-train or fit_reference and re-save."
        )

    cfg = ckpt.get("config") or {}
    img = img_override or cfg.get("img") or cfg.get("image_size") or frozen.get("image_size") or 256
    ldim = ldim_override or cfg.get("ldim") or frozen.get("latent_dim") or 100

    generator = CVAEGenerator_v2(latent_dim=ldim, image_size=img).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    mu_ref = ckpt["mu_ref"].to(device)
    Sigma_ref = ckpt["Sigma_ref"].to(device)
    # Prefer checkpoint fields; fall back to frozen YAML (same numerical defaults)
    sz_mode = ckpt.get("sz_mode") or cfg.get("sz_mode") or frozen.get("sz_mode") or "mu_only"
    sz_sigma_t_max = ckpt.get("sz_sigma_t_max")
    if sz_sigma_t_max is None:
        sz_sigma_t_max = cfg.get("sz_sigma_t_max", frozen.get("sz_sigma_t_max", 1.0))
    cfg = dict(cfg)
    cfg["sz_mode"] = sz_mode
    cfg["sz_sigma_t_max"] = float(sz_sigma_t_max)
    cfg["scoring_yaml"] = scoring_yaml or "configs/scoring_frozen.yaml"
    return generator, mu_ref, Sigma_ref, img, ldim, cfg


def load_image_tensor(path, img_size, device):
    image = Image.open(path).convert("RGB")
    image = _resize_short_side(image, img_size)
    image = center_crop(image, img_size)
    arr = (np.asarray(image) / 255.0).astype("float32")
    t = transforms.ToTensor()(arr).unsqueeze(0).to(device)
    return t, image


def collect_paths(folder):
    paths = []
    for dirpath, _, fns in os.walk(folder):
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    return sorted(paths)


def load_mos_map(csv_path, name_col, mos_col, has_header):
    mos = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) <= max(name_col, mos_col):
                continue
            mos[row[name_col]] = float(row[mos_col])
            mos[os.path.basename(row[name_col])] = float(row[mos_col])
    return mos


def main():
    opt = get_args()
    metrics = [m.strip().lower() for m in opt.metrics.split(",") if m.strip()]
    device = pick_device(opt)

    print(f"[INFO] Loading checkpoint {opt.ckpt}")
    print("[INFO] S_Z orientation: HIGHER S_Z = WORSE quality")
    generator, mu_ref, Sigma_ref, img, ldim, cfg = load_best(
        opt.ckpt, device, opt.img, opt.ldim, scoring_yaml=opt.scoring_yaml
    )
    print(f"[INFO] img={img} ldim={ldim}")

    if opt.image:
        paths = [opt.image]
    elif opt.folder:
        paths = collect_paths(opt.folder)
    else:
        print("[ERROR] provide --image or --folder")
        sys.exit(1)

    os.makedirs(opt.out_dir, exist_ok=True)
    transform_inv = transforms.ToPILImage()
    results = []

    mos_map = None
    if opt.mos_csv:
        mos_map = load_mos_map(opt.mos_csv, opt.name_col, opt.mos_col, opt.csv_has_header)

    for i, path in enumerate(paths):
        x, pil = load_image_tensor(path, img, device)
        row = {"path": path, "name": os.path.basename(path)}

        if "sz" in metrics:
            sz = score_sz_eval(
                generator,
                x,
                mu_ref,
                Sigma_ref,
                sigma_t_max=cfg.get("sz_sigma_t_max", 1.0),
                mu_only=(cfg.get("sz_mode", "mu_only") == "mu_only"),
            )
            row["sz"] = float(sz.view(-1)[0].item())
        if "rem" in metrics:
            rem = score_rem(generator, x)
            row["rem"] = float(rem.view(-1)[0].item())

        if opt.save_recon:
            with torch.no_grad():
                recon, _, _ = generator(x)
            rec_pil = transform_inv(recon[0].cpu().clamp(0, 1))
            outp = os.path.join(opt.out_dir, os.path.splitext(row["name"])[0] + "_recon.png")
            save_intermediateResults(pil, rec_pil, outp)

        msg = f"[{i+1}/{len(paths)}] {row['name']}"
        for k in ("sz", "rem"):
            if k in row:
                msg += f"  {k.upper()}={row[k]:.6f}"
        print(msg, end="\r")
        results.append(row)

    print()
    out_csv = os.path.join(opt.out_dir, "scores.csv")
    keys = ["name", "path"] + [k for k in ("sz", "rem") if k in metrics]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[INFO] wrote {out_csv}")

    if mos_map is not None and "sz" in metrics:
        y_true, y_pred = [], []
        for r in results:
            name = r["name"]
            if name in mos_map:
                y_true.append(mos_map[name])
                y_pred.append(r["sz"])
        if len(y_true) >= 3:
            # MOS usually higher=better; S_Z higher=worse -> expect NEGATIVE correlation
            srcc, _ = spearmanr(y_true, y_pred)
            plcc, _ = pearsonr(y_true, y_pred)
            krocc, _ = kendalltau(y_true, y_pred)
            print(f"[INFO] matched {len(y_true)} images with MOS")
            print(f"[INFO] S_Z vs MOS: SROCC={srcc:.4f}  PLCC={plcc:.4f}  KROCC={krocc:.4f}")
            print("[INFO] (expect negative corr if MOS higher=better, since S_Z higher=worse)")
        else:
            print(f"[WARN] only {len(y_true)} MOS matches; need >=3 for correlations")


if __name__ == "__main__":
    main()
