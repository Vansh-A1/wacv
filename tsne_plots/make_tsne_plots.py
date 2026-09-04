#!/usr/bin/env python3
"""Make requested t-SNE plots for E_A epoch 5 and frozen E_H on the same val set."""

from __future__ import annotations

import json
import sys
import time
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
OUT_DIR = ROOT / "tsne_plots"
CACHE_DIR = OUT_DIR / "cache"

EA_IDS = ROOT / "dataset_model2/day5_debug_eval/day9/day9_taskb/encoded_mu_cache/ep05_val_kadid_tid_ids.csv"
EA_MU = ROOT / "dataset_model2/day5_debug_eval/day9/day9_taskb/encoded_mu_cache/ep05_val_kadid_tid_mu.npy"
EH_CKPT = ROOT / "checkpoints/hr_combined_ft1/best.pth"

EH_IDS = CACHE_DIR / "eh_frozen_val_kadid_tid_ids.csv"
EH_MU = CACHE_DIR / "eh_frozen_val_kadid_tid_mu.npy"

SEED = 42
PERPLEXITY = 30.0
MAX_ITER = 1000
INIT = "pca"
LEARNING_RATE = "auto"

PATH_REMAPS = {
    "/data/projectwork/swati_mam/kadid10k": "/data/projectwork/swati_mam/new model data/kadid10k",
}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external"))
from external.dataloader import _resize_short_side, center_crop  # noqa: E402
from external.model import CVAEGenerator_v2  # noqa: E402


def remap_path(path: str) -> str:
    for old, new in PATH_REMAPS.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        path = str(self.df.loc[idx, "distorted_path"])
        pil = Image.open(path).convert("RGB")
        pil = _resize_short_side(pil, self.img_size)
        pil = center_crop(pil, self.img_size)
        arr = (np.asarray(pil) / 255.0).astype("float32")
        return self.to_tensor(arr), idx


def load_encoder(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    return model, img_size, latent_dim


@torch.no_grad()
def encode_eh_if_needed(ids: pd.DataFrame) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if EH_MU.exists() and EH_IDS.exists():
        cached = pd.read_csv(EH_IDS)
        if cached["image_id"].astype(str).tolist() == ids["image_id"].astype(str).tolist():
            return np.load(EH_MU)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, img_size, latent_dim = load_encoder(EH_CKPT, device)
    df = ids.copy()
    df["distorted_path"] = df["distorted_path"].astype(str).map(remap_path)
    missing = [p for p in df["distorted_path"].tolist() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image paths. First: {missing[0]}")

    batch_size = 64 if device.type == "cuda" else 16
    workers = 4 if device.type == "cuda" else 0
    loader = DataLoader(
        ImageDataset(df, img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda" and workers > 0),
    )
    mu = np.zeros((len(df), latent_dim), dtype=np.float32)
    for imgs, idxs in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        _recon, mu_batch, _logvar = model(imgs)
        mu[idxs.numpy()] = mu_batch.detach().cpu().numpy().astype(np.float32)
    np.save(EH_MU, mu)
    ids.to_csv(EH_IDS, index=False)
    return mu


def coarse_group(distortion_type: str) -> str:
    text = str(distortion_type).strip().lower()
    if "blur" in text or text in {"type_08", "type_23"}:
        return "blur"
    if "sharpen" in text or text == "type_05":
        return "HF/sharpen"
    if "noise" in text or text in {"impulse noise", "multiplicative noise", "type_01", "type_02", "type_03", "type_04", "type_06", "type_07", "type_19", "type_20"}:
        return "noise"
    return "other"


def compute_tsne(mu: np.ndarray, label: str) -> np.ndarray:
    coords_path = CACHE_DIR / f"tsne_{label}_seed{SEED}_perp{int(PERPLEXITY)}.npy"
    if coords_path.exists():
        return np.load(coords_path)
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        random_state=SEED,
        init=INIT,
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITER,
        metric="euclidean",
    )
    coords = tsne.fit_transform(mu).astype(np.float32)
    np.save(coords_path, coords)
    return coords


def plot_by_column(df: pd.DataFrame, x: str, y: str, color_col: str, title: str, out_path: Path) -> None:
    values = sorted(df[color_col].astype(str).unique())
    cmap = plt.get_cmap("tab20", max(20, len(values)))
    color_map = {v: cmap(i % cmap.N) for i, v in enumerate(values)}

    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
    for value in values:
        sub = df[df[color_col].astype(str).eq(value)]
        ax.scatter(sub[x], sub[y], s=14, alpha=0.78, color=color_map[value], label=value, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False, markerscale=1.4)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(EA_IDS)
    ea_mu = np.load(EA_MU)
    if len(ids) != len(ea_mu):
        raise ValueError(f"E_A ids rows {len(ids)} != mu rows {len(ea_mu)}")
    if ids["dataset"].eq("KONIQ-10k").any():
        raise RuntimeError("Unexpected KonIQ rows in the requested val-set")

    started = time.time()
    eh_mu = encode_eh_if_needed(ids)
    if eh_mu.shape != ea_mu.shape:
        raise ValueError(f"E_H mu shape {eh_mu.shape} != E_A mu shape {ea_mu.shape}")

    ids = ids.copy()
    ids["group_B"] = ids["distortion_type"].map(coarse_group)
    ea_xy = compute_tsne(ea_mu, "EA_epoch5_val_kadid_tid")
    eh_xy = compute_tsne(eh_mu, "EH_frozen_val_kadid_tid")

    out = ids.copy()
    out["EA_tsne_x"] = ea_xy[:, 0]
    out["EA_tsne_y"] = ea_xy[:, 1]
    out["EH_tsne_x"] = eh_xy[:, 0]
    out["EH_tsne_y"] = eh_xy[:, 1]
    out.to_csv(OUT_DIR / "tsne_coordinates_same_valset.csv", index=False)

    plots = {
        "01_EA_epoch5_A_by_distortion_type.png": ("EA_tsne_x", "EA_tsne_y", "distortion_type", "E_A epoch 5 t-SNE, colored by distortion type"),
        "02_EA_epoch5_B_by_group.png": ("EA_tsne_x", "EA_tsne_y", "group_B", "E_A epoch 5 t-SNE, colored by group B"),
        "03_EH_frozen_A_by_distortion_type.png": ("EH_tsne_x", "EH_tsne_y", "distortion_type", "Frozen E_H t-SNE, colored by distortion type"),
        "04_EH_frozen_B_by_group.png": ("EH_tsne_x", "EH_tsne_y", "group_B", "Frozen E_H t-SNE, colored by group B"),
    }
    for filename, (x, y, color_col, title) in plots.items():
        plot_by_column(out, x, y, color_col, title, OUT_DIR / filename)

    summary = {
        "task": "t-SNE plots for same validation set",
        "output_dir": str(OUT_DIR),
        "same_val_set": str(EA_IDS),
        "N": int(len(out)),
        "dataset_counts": {str(k): int(v) for k, v in out["dataset"].value_counts().to_dict().items()},
        "EA_mu_source": str(EA_MU),
        "EH_mu_source": str(EH_MU),
        "EH_checkpoint": str(EH_CKPT),
        "tsne_settings": {
            "seed": SEED,
            "perplexity": PERPLEXITY,
            "max_iter": MAX_ITER,
            "init": INIT,
            "learning_rate": LEARNING_RATE,
            "metric": "euclidean",
        },
        "group_B_definition": {
            "blur": "distortion name contains blur, plus TID type_08/type_23",
            "HF/sharpen": "High sharpen/sharpen names, plus TID type_05",
            "noise": "noise names plus TID noise-like type codes",
            "other": "all remaining types",
        },
        "group_counts": {str(k): int(v) for k, v in out["group_B"].value_counts().to_dict().items()},
        "plots": [str(OUT_DIR / name) for name in plots],
        "coordinates_csv": str(OUT_DIR / "tsne_coordinates_same_valset.csv"),
        "runtime_seconds": float(time.time() - started),
    }
    (OUT_DIR / "tsne_summary.json").write_text(json.dumps(summary, indent=2))

    report = [
        "t-SNE plot task completed",
        "",
        "Four plots were made on the same validation set with identical t-SNE settings.",
        f"Validation set: {EA_IDS}",
        f"N: {len(out)}",
        f"Dataset counts: {summary['dataset_counts']}",
        f"Group B counts: {summary['group_counts']}",
        "",
        "t-SNE settings:",
        json.dumps(summary["tsne_settings"], indent=2),
        "",
        "Outputs:",
    ]
    report.extend(summary["plots"])
    report.append(str(OUT_DIR / "tsne_coordinates_same_valset.csv"))
    (OUT_DIR / "README_tsne_plots.txt").write_text("\n".join(report) + "\n")
    print(f"Done: {OUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
