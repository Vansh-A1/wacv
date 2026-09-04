#!/usr/bin/env python3
"""Day 9 Task A: encode train mu vectors and fit diagonal GMMs."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PACKAGE_ROOT = Path("/home/projectwork/student_package")
DATASET_ROOT = PACKAGE_ROOT / "dataset_model2"
OUT_ROOT = DATASET_ROOT / "day5_debug_eval" / "day9" / "day9_taskA"
GMM_ROOT = OUT_ROOT / "day9_gmm"

SPLIT_CSV = DATASET_ROOT / "ref_splits_seed42" / "combined_kadid_tid_koniq_split_seed42.csv"
CKPT_DIR = DATASET_ROOT / "checkpoints" / "wacv_ea_day7_seed42"
CHECKPOINTS = {
    "ep05": CKPT_DIR / "epoch_0005.pth",
    "ep10": CKPT_DIR / "epoch_0010.pth",
}
K_VALUES = [2, 4, 8, 16]
SEED = 42

PATH_REMAPS = {
    "/data/projectwork/swati_mam/kadid10k": "/data/projectwork/swati_mam/new model data/kadid10k",
}

sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "external"))
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


def load_train_manifest() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    train = df[df["split"].eq("train")].copy().reset_index(drop=True)
    train["distorted_path_original"] = train["distorted_path"]
    train["distorted_path"] = train["distorted_path"].astype(str).map(remap_path)
    missing = [p for p in train["distorted_path"].tolist() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} train images. First missing path: {missing[0]}")
    return train


def load_encoder(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    meta = {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "img_size": img_size,
        "latent_dim": latent_dim,
        "has_mu_ref": "mu_ref" in ckpt,
        "has_Sigma_ref": "Sigma_ref" in ckpt,
    }
    return model, img_size, latent_dim, meta


@torch.no_grad()
def encode_mu(df: pd.DataFrame, ckpt_path: Path, device: torch.device, out_dir: Path) -> tuple[np.ndarray, dict]:
    model, img_size, latent_dim, meta = load_encoder(ckpt_path, device)
    batch_size = 64 if device.type == "cuda" else 16
    workers = 4 if device.type == "cuda" else 0
    loader = DataLoader(
        ImageDataset(df, img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda" and workers > 0),
    )
    mu_arr = np.zeros((len(df), latent_dim), dtype=np.float32)
    started = time.time()
    for imgs, idxs in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        _recon, mu, _logvar = model(imgs)
        mu_arr[idxs.numpy()] = mu.detach().cpu().numpy().astype(np.float32)
    meta.update(
        {
            "train_N": int(len(df)),
            "mu_shape": list(mu_arr.shape),
            "mu_dtype": str(mu_arr.dtype),
            "encode_seconds": float(time.time() - started),
            "batch_size": batch_size,
            "workers": workers,
            "device": str(device),
        }
    )
    np.save(out_dir / "mu_train.npy", mu_arr)
    return mu_arr, meta


def save_id_list(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "dataset",
        "image_id",
        "ref_id",
        "distortion_type",
        "distortion_code",
        "severity_or_level",
        "mos_or_dmos",
        "split",
        "distorted_path",
        "distorted_path_original",
        "ref_path",
    ]
    present = [c for c in cols if c in df.columns]
    df[present].to_csv(out_dir / "mu_train_ids.csv", index=False)


def fit_and_save_gmms(mu_arr: np.ndarray, out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for k in K_VALUES:
        started = time.time()
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=SEED,
            reg_covar=1e-6,
            max_iter=300,
            n_init=3,
            init_params="kmeans",
        )
        gmm.fit(mu_arr)
        model_dict = {
            "model_type": "sklearn.mixture.GaussianMixture",
            "sklearn_version": __import__("sklearn").__version__,
            "covariance_type": "diag",
            "K": int(k),
            "seed": SEED,
            "train_N": int(mu_arr.shape[0]),
            "latent_dim": int(mu_arr.shape[1]),
            "weights": torch.tensor(gmm.weights_, dtype=torch.float32),
            "means": torch.tensor(gmm.means_, dtype=torch.float32),
            "covariances": torch.tensor(gmm.covariances_, dtype=torch.float32),
            "precisions_cholesky": torch.tensor(gmm.precisions_cholesky_, dtype=torch.float32),
            "lower_bound": float(gmm.lower_bound_),
            "n_iter": int(gmm.n_iter_),
            "converged": bool(gmm.converged_),
            "reg_covar": float(gmm.reg_covar),
            "max_iter": int(gmm.max_iter),
            "n_init": int(gmm.n_init),
            "init_params": str(gmm.init_params),
        }
        torch.save(model_dict, out_dir / f"gmm_K{k:02d}.pt")
        rows.append(
            {
                "K": k,
                "model_path": str(out_dir / f"gmm_K{k:02d}.pt"),
                "train_N": int(mu_arr.shape[0]),
                "latent_dim": int(mu_arr.shape[1]),
                "seed": SEED,
                "covariance_type": "diag",
                "converged": bool(gmm.converged_),
                "n_iter": int(gmm.n_iter_),
                "lower_bound": float(gmm.lower_bound_),
                "fit_seconds": float(time.time() - started),
            }
        )
    return rows


def run_epoch(tag: str, ckpt_path: Path, train_df: pd.DataFrame, device: torch.device) -> dict:
    out_dir = GMM_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    save_id_list(train_df, out_dir)
    mu_arr, enc_meta = encode_mu(train_df, ckpt_path, device, out_dir)
    fit_rows = fit_and_save_gmms(mu_arr, out_dir)
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(out_dir / "fit_log.csv", index=False)

    lines = [
        f"Day 9 Task A GMM fit log for {tag}",
        f"checkpoint: {ckpt_path}",
        f"device: {device}",
        f"train_N: {len(train_df)}",
        f"mu_train.npy shape: {mu_arr.shape}",
        f"seed: {SEED}",
        "GMM settings: sklearn GaussianMixture, covariance_type=diag, reg_covar=1e-6, max_iter=300, n_init=3, init_params=kmeans",
        "",
        fit_df.to_string(index=False),
        "",
    ]
    (out_dir / "fit_log.txt").write_text("\n".join(lines))
    done_lines = [
        f"{tag} DONE",
        f"mu_train.npy: {out_dir / 'mu_train.npy'}",
        f"mu_train_ids.csv: {out_dir / 'mu_train_ids.csv'}",
    ]
    done_lines.extend([f"gmm_K{k:02d}.pt: {out_dir / f'gmm_K{k:02d}.pt'}" for k in K_VALUES])
    (out_dir / "DONE.txt").write_text("\n".join(done_lines) + "\n")
    return {
        "tag": tag,
        "checkpoint": str(ckpt_path),
        "output_dir": str(out_dir),
        "encoding": enc_meta,
        "gmms": fit_rows,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    GMM_ROOT.mkdir(parents=True, exist_ok=True)
    for ckpt in CHECKPOINTS.values():
        if not ckpt.is_file():
            raise FileNotFoundError(str(ckpt))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    train_df = load_train_manifest()
    train_df.to_csv(OUT_ROOT / "day9_taskA_train_manifest_used.csv", index=False)

    all_results = []
    for tag, ckpt_path in CHECKPOINTS.items():
        all_results.append(run_epoch(tag, ckpt_path, train_df, device))

    sync_rows = []
    for result in all_results:
        for gmm in result["gmms"]:
            sync_rows.append(
                {
                    "epoch_tag": result["tag"],
                    "checkpoint": result["checkpoint"],
                    "train_N": gmm["train_N"],
                    "K": gmm["K"],
                    "model_path": gmm["model_path"],
                }
            )
    sync_df = pd.DataFrame(sync_rows)
    sync_df.to_csv(OUT_ROOT / "day9_taskA_sync.csv", index=False)

    summary = {
        "task": "Day 9 Task A - encode train mu and fit diagonal GMMs",
        "output_root": str(OUT_ROOT),
        "gmm_root": str(GMM_ROOT),
        "split_csv": str(SPLIT_CSV),
        "seed": SEED,
        "full_train_used": True,
        "train_N": int(len(train_df)),
        "train_counts_by_dataset": {str(k): int(v) for k, v in train_df["dataset"].value_counts().to_dict().items()},
        "epochs": all_results,
        "sync_csv": str(OUT_ROOT / "day9_taskA_sync.csv"),
    }
    (OUT_ROOT / "day9_taskA_summary.json").write_text(json.dumps(summary, indent=2))

    report = [
        "Day 9 Task A completed",
        "",
        "Full train split was used, not a subset.",
        f"Train N: {len(train_df)}",
        f"Device used: {device}",
        f"Seed: {SEED}",
        "",
        "Saved per epoch under:",
        str(GMM_ROOT),
        "",
        "Sync:",
        sync_df.to_string(index=False),
        "",
    ]
    (OUT_ROOT / "day9_taskA_report.txt").write_text("\n".join(report))
    (OUT_ROOT / "DONE.txt").write_text(
        "Day 9 Task A DONE\n"
        f"ep05: {GMM_ROOT / 'ep05'}\n"
        f"ep10: {GMM_ROOT / 'ep10'}\n"
        f"sync: {OUT_ROOT / 'day9_taskA_sync.csv'}\n"
    )

    print(f"Day 9 Task A complete: {OUT_ROOT}")
    print(sync_df.to_string(index=False))


if __name__ == "__main__":
    main()
