#!/usr/bin/env python3
"""Create t-SNE plots for available Day11 and Task12 checkpoints.

The script uses the same KADID+TID validation split, excludes KonIQ, encodes
posterior mean vectors, and writes plots into each run's own latent_plots folder.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path("/home/projectwork/student_package")
DAY_ROOT = ROOT / "day11_day12"
SPLIT_CSV = ROOT / "dataset_model2/ref_splits_seed42/combined_kadid_tid_koniq_split_seed42.csv"
SEED = 42
PERPLEXITY = 30.0
MAX_ITER = 1000
MAX_IMAGES = 1000

PATH_REMAPS = {
    "/data/projectwork/swati_mam/kadid10k": "/data/projectwork/swati_mam/new model data/kadid10k",
}

sys.path.insert(0, str(ROOT))
from external.dataloader import _resize_short_side, center_crop  # noqa: E402
from external.model import CVAEGenerator_v2  # noqa: E402


def remap_path(path: str) -> str:
    for old, new in PATH_REMAPS.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


def coarse_group(distortion_type: str) -> str:
    text = str(distortion_type).lower()
    if "blur" in text or text in {"type_08", "type_23"}:
        return "blur"
    if "sharpen" in text or text == "type_05":
        return "HF/sharpen"
    if "noise" in text or text in {"type_01", "type_02", "type_03", "type_04", "type_06", "type_07", "type_19", "type_20"}:
        return "noise"
    return "other"


def load_val_ids() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    df = df[df["split"].eq("val") & df["dataset"].isin(["KADID-10k", "TID2013"])].copy()
    df["severity_or_level"] = pd.to_numeric(df["severity_or_level"], errors="coerce")
    df["distorted_path"] = df["distorted_path"].map(remap_path)
    df = df.sort_values(["dataset", "ref_id", "distortion_type", "severity_or_level", "image_id"]).reset_index(drop=True)
    if len(df) > MAX_IMAGES:
        df = df.sample(n=MAX_IMAGES, random_state=SEED).sort_values(["dataset", "ref_id", "distortion_type", "severity_or_level", "image_id"]).reset_index(drop=True)
    if df["dataset"].eq("KONIQ-10k").any():
        raise RuntimeError("KonIQ leaked into t-SNE validation set")
    missing = [p for p in df["distorted_path"].tolist() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image path: {missing[0]}")
    df["group"] = df["distortion_type"].map(coarse_group)
    return df


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        pil = Image.open(str(self.df.loc[idx, "distorted_path"])).convert("RGB")
        pil = _resize_short_side(pil, self.img_size)
        pil = center_crop(pil, self.img_size)
        arr = (np.asarray(pil) / 255.0).astype("float32")
        return self.to_tensor(arr), idx


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    return model, img_size, latent_dim, ckpt


@torch.no_grad()
def encode_mu(df: pd.DataFrame, ckpt_path: Path, cache_dir: Path, device: torch.device) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    label = ckpt_path.stem
    mu_path = cache_dir / f"{label}_mu.npy"
    ids_path = cache_dir / f"{label}_ids.csv"
    if mu_path.exists() and ids_path.exists():
        cached = pd.read_csv(ids_path)
        if cached["image_id"].astype(str).tolist() == df["image_id"].astype(str).tolist():
            return np.load(mu_path)

    model, img_size, latent_dim, _ckpt = load_model(ckpt_path, device)
    workers = 0 if device.type == "cpu" else 4
    batch_size = 16 if device.type == "cpu" else 64
    loader = DataLoader(ImageDataset(df, img_size), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    mu = np.zeros((len(df), latent_dim), dtype=np.float32)
    for imgs, idxs in loader:
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        _recon, mu_batch, _logvar = model(imgs)
        mu[idxs.numpy()] = mu_batch.detach().cpu().numpy().astype(np.float32)
    np.save(mu_path, mu)
    df[["dataset", "image_id", "ref_id", "distortion_type", "severity_or_level", "distorted_path", "group"]].to_csv(ids_path, index=False)
    return mu


def make_tsne(mu: np.ndarray, cache_dir: Path, label: str) -> np.ndarray:
    out = cache_dir / f"{label}_tsne_seed{SEED}_perp{int(PERPLEXITY)}.npy"
    if out.exists():
        return np.load(out)
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
        max_iter=MAX_ITER,
        metric="euclidean",
    )
    xy = tsne.fit_transform(mu).astype(np.float32)
    np.save(out, xy)
    return xy


def plot_scatter(frame: pd.DataFrame, color_col: str, title: str, out_path: Path) -> None:
    values = sorted(frame[color_col].astype(str).unique())
    cmap = plt.get_cmap("tab20", max(20, len(values)))
    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
    for i, value in enumerate(values):
        sub = frame[frame[color_col].astype(str).eq(value)]
        ax.scatter(sub["tsne_x"], sub["tsne_y"], s=14, alpha=0.78, linewidths=0, color=cmap(i % cmap.N), label=value)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False, markerscale=1.4)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(out_path)
    plt.close(fig)


def epoch_from_path(path: Path) -> int:
    m = re.search(r"epoch_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def run_one(run_name: str, run_dir: Path, df: pd.DataFrame, device: torch.device) -> dict:
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch_*.pth"), key=epoch_from_path)
    latent_dir = run_dir / "latent_plots"
    cache_dir = latent_dir / "cache"
    latent_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ckpt_path in ckpts:
        epoch = epoch_from_path(ckpt_path)
        label = f"epoch_{epoch:04d}"
        mu = encode_mu(df, ckpt_path, cache_dir, device)
        xy = make_tsne(mu, cache_dir, label)
        frame = df.copy()
        frame["tsne_x"] = xy[:, 0]
        frame["tsne_y"] = xy[:, 1]
        coords = latent_dir / f"{label}_tsne_coordinates.csv"
        frame.to_csv(coords, index=False)
        by_type = latent_dir / f"{label}_tsne_by_distortion_type.png"
        by_group = latent_dir / f"{label}_tsne_by_group.png"
        plot_scatter(frame, "distortion_type", f"{run_name} {label}: by distortion type", by_type)
        plot_scatter(frame, "group", f"{run_name} {label}: by grouped type", by_group)
        outputs.extend([str(coords), str(by_type), str(by_group)])
    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "checkpoint_count": len(ckpts),
        "checkpoints": [str(p) for p in ckpts],
        "outputs": outputs,
    }
    (latent_dir / "tsne_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    df = load_val_ids()
    runs = [
        ("Day11_rankall_lambda_0.1", DAY_ROOT / "day11_rankall/lambda_0.1"),
        ("Day11_rankall_lambda_1.0", DAY_ROOT / "day11_rankall/lambda_1.0"),
        ("Task12_pristine_rank_lambda_0.1", DAY_ROOT / "task12/day11b_pristine_rank/lambda_0.1"),
        ("Task12_pristine_rank_lambda_1.0", DAY_ROOT / "task12/day11b_pristine_rank/lambda_1.0"),
    ]
    summaries = []
    for name, run_dir in runs:
        if (run_dir / "checkpoints").is_dir() and list((run_dir / "checkpoints").glob("epoch_*.pth")):
            summaries.append(run_one(name, run_dir, df, device))
        else:
            summaries.append({"run_name": name, "run_dir": str(run_dir), "checkpoint_count": 0, "skipped": "no saved epoch_*.pth checkpoints found"})
    full = {
        "output_root": str(DAY_ROOT),
        "N": int(len(df)),
        "max_images": MAX_IMAGES,
        "datasets": {str(k): int(v) for k, v in df["dataset"].value_counts().to_dict().items()},
        "tsne_settings": {"seed": SEED, "perplexity": PERPLEXITY, "max_iter": MAX_ITER, "init": "pca", "learning_rate": "auto"},
        "runs": summaries,
    }
    (DAY_ROOT / "day11_day12_tsne_summary.json").write_text(json.dumps(full, indent=2) + "\n")
    print(json.dumps(full, indent=2))


if __name__ == "__main__":
    main()
