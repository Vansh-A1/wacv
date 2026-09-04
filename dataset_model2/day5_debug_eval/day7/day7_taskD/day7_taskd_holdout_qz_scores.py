#!/usr/bin/env python3
"""Day 7 Task D: validation-normalized holdout scores using selected E_A checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PACKAGE_ROOT = Path("/home/projectwork/student_package")
DATASET_ROOT = PACKAGE_ROOT / "dataset_model2"
DAY5_DIR = DATASET_ROOT / "day5_debug_eval"
DAY7_DIR = DAY5_DIR / "day7"
OUT_DIR = DAY7_DIR / "day7_taske"

SPLIT_CSV = DATASET_ROOT / "ref_splits_seed42" / "combined_kadid_tid_koniq_split_seed42.csv"
TASKC_SELECTION = DAY7_DIR / "day7_taskd" / "taskc_checkpoint_selection_table.csv"
EH_CKPT = PACKAGE_ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"

KONIQ_MOS_CANDIDATES = [
    DAY5_DIR / "day6" / "koniq10k_scores.csv",
    DAY5_DIR / "day6_task" / "day6" / "koniq10k_scores.csv",
]

OUT_VAL_SCORES = OUT_DIR / "taskD_val_scores.csv"
OUT_VAL_STATS = OUT_DIR / "taskD_val_normalization_stats.csv"
OUT_HOLDOUT = OUT_DIR / "taskD_holdout_scores.csv"
OUT_SUMMARY = OUT_DIR / "taskD_summary.json"
OUT_REPORT = OUT_DIR / "taskD_report.txt"

PATH_REMAPS = {
    "/data/projectwork/swati_mam/kadid10k": "/data/projectwork/swati_mam/new model data/kadid10k",
}

sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "external"))
from external.dataloader import _resize_short_side, center_crop  # noqa: E402
from external.model import CVAEGenerator_v2  # noqa: E402
from score import sz_from_stats  # noqa: E402


def remap_path(path: str) -> str:
    for old, new in PATH_REMAPS.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], img_size: int):
        self.paths = paths
        self.img_size = img_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        pil = Image.open(path).convert("RGB")
        pil = _resize_short_side(pil, self.img_size)
        pil = center_crop(pil, self.img_size)
        arr = (np.asarray(pil) / 255.0).astype("float32")
        return self.to_tensor(arr), idx


def selected_ea_checkpoint() -> Path:
    sel = pd.read_csv(TASKC_SELECTION)
    chosen = sel[sel["selected"].astype(str).str.lower() == "yes"]
    if len(chosen) != 1:
        raise ValueError(f"Expected exactly one selected checkpoint in {TASKC_SELECTION}, found {len(chosen)}")
    return Path(str(chosen.iloc[0]["ckpt path"]))


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    if "mu_ref" not in ckpt or "Sigma_ref" not in ckpt:
        raise KeyError(f"{ckpt_path} has no mu_ref/Sigma_ref")
    meta = {
        "checkpoint": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "img_size": img_size,
        "latent_dim": latent_dim,
        "mu_ref_shape": list(ckpt["mu_ref"].shape),
        "Sigma_ref_shape": list(ckpt["Sigma_ref"].shape),
        "Sigma_ref_mean": float(ckpt["Sigma_ref"].float().mean().item()),
        "sz_mode": ckpt.get("sz_mode", "mu_only"),
        "sz_sigma_t_max": float(ckpt.get("sz_sigma_t_max", 1.0)),
    }
    return model, ckpt["mu_ref"].to(device), ckpt["Sigma_ref"].to(device), img_size, meta


@torch.no_grad()
def score_paths(paths: list[str], model, mu_ref, sigma_ref, img_size: int, device: torch.device, batch_size: int, workers: int) -> list[float]:
    loader = DataLoader(
        ImagePathDataset(paths, img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda" and workers > 0),
    )
    scores: list[float] = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        _, mu, logvar = model(imgs)
        sz = sz_from_stats(mu, logvar, mu_ref, sigma_ref, sigma_t_max=1.0, mu_only=True)
        scores.extend(sz.detach().cpu().numpy().tolist())
    return scores


def load_koniq_mos() -> tuple[dict[str, float], str | None]:
    for path in KONIQ_MOS_CANDIDATES:
        if path.is_file():
            df = pd.read_csv(path)
            df["image_id"] = df["image_name"].map(lambda x: Path(str(x)).stem)
            return dict(zip(df["image_id"].astype(str), pd.to_numeric(df["MOS"], errors="coerce"))), str(path)
    return {}, None


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    df["distorted_path"] = df["distorted_path"].map(remap_path)
    df["severity_or_level"] = pd.to_numeric(df["severity_or_level"], errors="coerce")
    df["mos_or_dmos"] = pd.to_numeric(df["mos_or_dmos"], errors="coerce")

    koniq_mos, mos_source = load_koniq_mos()
    if koniq_mos:
        mask = df["dataset"].eq("KONIQ-10k")
        df.loc[mask, "mos_or_dmos"] = df.loc[mask, "image_id"].astype(str).map(koniq_mos)
    df.attrs["koniq_mos_source"] = mos_source
    return df


def score_frame(df: pd.DataFrame, model_h_pack, model_a_pack, device: torch.device, batch_size: int, workers: int) -> pd.DataFrame:
    paths = df["distorted_path"].astype(str).tolist()
    mh, muh, sigh, imgh, _ = model_h_pack
    ma, mua, siga, imga, _ = model_a_pack
    if imgh != imga:
        raise ValueError(f"E_H img size {imgh} != E_A img size {imga}")
    out = df.copy()
    out["E_H"] = score_paths(paths, mh, muh, sigh, imgh, device, batch_size, workers)
    out["E_A"] = score_paths(paths, ma, mua, siga, imga, device, batch_size, workers)
    return out


def val_stats(val_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in val_scores.groupby("dataset", sort=True):
        rows.append({
            "dataset": dataset,
            "N_val": int(len(g)),
            "mean_H": float(g["E_H"].mean()),
            "std_H": float(g["E_H"].std(ddof=0)),
            "mean_A": float(g["E_A"].mean()),
            "std_A": float(g["E_A"].std(ddof=0)),
        })
    return pd.DataFrame(rows)


def apply_stats(holdout: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    st = stats.set_index("dataset")
    rows = []
    for _, row in holdout.iterrows():
        dataset = row["dataset"]
        if dataset not in st.index:
            continue
        mean_h = float(st.loc[dataset, "mean_H"])
        std_h = float(st.loc[dataset, "std_H"])
        mean_a = float(st.loc[dataset, "mean_A"])
        std_a = float(st.loc[dataset, "std_A"])
        z_h = (float(row["E_H"]) - mean_h) / std_h if std_h > 0 else np.nan
        z_a = (float(row["E_A"]) - mean_a) / std_a if std_a > 0 else np.nan
        out = row.to_dict()
        out["Q_raw"] = float(row["E_A"]) - float(row["E_H"])
        out["z_H"] = z_h
        out["z_A"] = z_a
        out["Qz_AmH"] = z_a - z_h
        out["Qz_HmA"] = z_h - z_a
        rows.append(out)
    cols = [
        "dataset", "image_id", "ref_id", "distortion_type", "severity_or_level",
        "mos_or_dmos", "E_H", "E_A", "Q_raw", "z_H", "z_A", "Qz_AmH", "Qz_HmA",
    ]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="allow CPU run for debugging only")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Run this from the GPU terminal, or pass --cpu only for debugging.")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    ea_ckpt = selected_ea_checkpoint()
    model_h_pack = load_model(EH_CKPT, device)
    model_a_pack = load_model(ea_ckpt, device)

    manifest = load_manifest()
    val = manifest[manifest["split"].eq("val") & manifest["dataset"].isin(["KADID-10k", "TID2013", "KONIQ-10k"])].copy()
    holdout = manifest[manifest["split"].eq("holdout") & manifest["dataset"].isin(["KADID-10k", "TID2013", "KONIQ-10k"])].copy()

    val_scored = score_frame(val, model_h_pack, model_a_pack, device, args.batch_size, args.workers)
    val_scored.to_csv(OUT_VAL_SCORES, index=False)

    stats = val_stats(val_scored)
    stats.to_csv(OUT_VAL_STATS, index=False)

    holdout_scored = score_frame(holdout, model_h_pack, model_a_pack, device, args.batch_size, args.workers)
    taskd = apply_stats(holdout_scored, stats)
    taskd.to_csv(OUT_HOLDOUT, index=False)

    _, _, _, _, eh_meta = model_h_pack
    _, _, _, _, ea_meta = model_a_pack
    summary = {
        "task": "Day 7 Task D: validation-normalized holdout Q scores",
        "output_dir": str(OUT_DIR),
        "device": str(device),
        "frozen_pristine_E_H": eh_meta,
        "selected_artifact_E_A": ea_meta,
        "taskC_selection_csv": str(TASKC_SELECTION),
        "selected_E_A_checkpoint": str(ea_ckpt),
        "E_A_reference_source": "embedded mu_ref/Sigma_ref inside selected Task C checkpoint",
        "koniq_mos_source": manifest.attrs.get("koniq_mos_source"),
        "val_counts_by_dataset": val["dataset"].value_counts().to_dict(),
        "holdout_counts_by_dataset": holdout["dataset"].value_counts().to_dict(),
        "normalization_rule": "For each dataset separately, mean/std of E_H and E_A are fit on val only. Holdout z-scores use those val-only stats.",
        "formulas": {
            "E_H": "pristine energy",
            "E_A": "artifact energy using selected E_A checkpoint and new ref",
            "Q_raw": "E_A - E_H",
            "z_H": "(E_H - mean_H_val_dataset) / std_H_val_dataset",
            "z_A": "(E_A - mean_A_val_dataset) / std_A_val_dataset",
            "Qz_AmH": "z_A - z_H",
            "Qz_HmA": "z_H - z_A",
        },
        "outputs": {
            "taskD_holdout_scores_csv": str(OUT_HOLDOUT),
            "taskD_val_scores_csv": str(OUT_VAL_SCORES),
            "taskD_val_normalization_stats_csv": str(OUT_VAL_STATS),
            "summary_json": str(OUT_SUMMARY),
            "report_txt": str(OUT_REPORT),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "Day 7 Task D: validation-normalized holdout scores",
        "",
        f"Frozen pristine E_H checkpoint: {EH_CKPT}",
        f"Selected artifact E_A checkpoint: {ea_ckpt}",
        f"E_A ref source: embedded new ref in selected checkpoint, Sigma mean={ea_meta['Sigma_ref_mean']:.12g}",
        f"Device: {device}",
        "",
        "Val-only normalization stats:",
        stats.to_string(index=False),
        "",
        "Holdout output columns:",
        ", ".join(taskd.columns),
        "",
        f"Holdout rows saved: {len(taskd)}",
        f"Holdout by dataset: {taskd['dataset'].value_counts().to_dict()}",
        "",
        "Outputs:",
        f"- {OUT_HOLDOUT}",
        f"- {OUT_VAL_SCORES}",
        f"- {OUT_VAL_STATS}",
        f"- {OUT_SUMMARY}",
        f"- {OUT_REPORT}",
    ]
    OUT_REPORT.write_text("\n".join(report) + "\n")
    print(f"Saved Task D outputs under {OUT_DIR}")
    print(f"Saved holdout CSV: {OUT_HOLDOUT}")


if __name__ == "__main__":
    main()
