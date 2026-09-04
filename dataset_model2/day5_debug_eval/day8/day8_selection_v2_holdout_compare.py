#!/usr/bin/env python3
"""Day 8: recompute Task C v2 selection and compare epoch 5 vs v2 on KADID holdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PACKAGE_ROOT = Path("/home/projectwork/student_package")
DATASET_ROOT = PACKAGE_ROOT / "dataset_model2"
DAY5_ROOT = DATASET_ROOT / "day5_debug_eval"
DAY7_ROOT = DAY5_ROOT / "day7"
OUT_DIR = DATASET_ROOT / "day8"

TASKC_SELECTION = DAY7_ROOT / "day7_taskc" / "taskc_checkpoint_selection_table.csv"
TASKC_VAL_SCORES = DAY7_ROOT / "day7_taskc" / "taskc_val_scores_all_epoch_ckpts.csv"
TASKD_HOLDOUT = DAY7_ROOT / "day7_taskD" / "taskD_holdout_scores.csv"
TASKD_VAL_STATS = DAY7_ROOT / "day7_taskD" / "taskD_val_normalization_stats.csv"
DAY5_HOLDOUT = DAY5_ROOT / "day5_holdout_scores.csv"

V1_EPOCH = 5
KADID = "KADID-10k"

OUT_SELECTION = OUT_DIR / "taskC_v2_checkpoint_selection_table.csv"
OUT_V2_HOLDOUT = OUT_DIR / "kadid_holdout_v2_selected_epoch_scores.csv"
OUT_COMPARE_SCORES = OUT_DIR / "kadid_holdout_epoch5_v1_vs_v2_scores.csv"
OUT_COMPARE_TABLE = OUT_DIR / "kadid_holdout_epoch5_v1_vs_v2_metric_compare.csv"
OUT_REPORT = OUT_DIR / "day8_report.txt"
OUT_SUMMARY = OUT_DIR / "day8_summary.json"
OUT_RUN_LOG = OUT_DIR / "day8_run_log.txt"

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
        pil = Image.open(self.paths[idx]).convert("RGB")
        pil = _resize_short_side(pil, self.img_size)
        pil = center_crop(pil, self.img_size)
        arr = (np.asarray(pil) / 255.0).astype("float32")
        return self.to_tensor(arr), idx


def finite_corr(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float | int]:
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {"N": n, "SRCC": np.nan, "SRCC_p": np.nan, "Pearson": np.nan, "Pearson_p": np.nan}
    srcc = spearmanr(x, y)
    pr = pearsonr(x, y)
    return {
        "N": n,
        "SRCC": float(srcc.statistic),
        "SRCC_p": float(srcc.pvalue),
        "Pearson": float(pr.statistic),
        "Pearson_p": float(pr.pvalue),
    }


def recompute_selection_v2() -> pd.DataFrame:
    sel = pd.read_csv(TASKC_SELECTION)
    out = sel.rename(
        columns={
            "blur SRCC": "blur",
            "lens SRCC": "lens",
            "sharpen SRCC": "sharpen",
            "pixelation SRCC": "pixelate",
            "pooled Spearman(E_A,MOS)": "pooled_EA_MOS_report_only",
            "selection_score": "selection_score_v1",
        }
    ).copy()
    out["selection_score_v2"] = (
        -pd.to_numeric(out["blur"], errors="coerce")
        - pd.to_numeric(out["lens"], errors="coerce")
        - np.maximum(0.0, pd.to_numeric(out["sharpen"], errors="coerce"))
        - np.maximum(0.0, pd.to_numeric(out["pixelate"], errors="coerce"))
    )
    best_idx = out["selection_score_v2"].idxmax()
    out["selected_v2"] = "no"
    out.loc[best_idx, "selected_v2"] = "yes"
    cols = [
        "Epoch",
        "ckpt path",
        "blur",
        "lens",
        "sharpen",
        "pixelate",
        "pooled_EA_MOS_report_only",
        "selection_score_v1",
        "selection_score_v2",
        "selected_v2",
    ]
    out[cols].to_csv(OUT_SELECTION, index=False)
    return out[cols]


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    if "mu_ref" not in ckpt or "Sigma_ref" not in ckpt:
        raise KeyError(f"{ckpt_path} is missing embedded mu_ref/Sigma_ref")
    return model, ckpt["mu_ref"].to(device), ckpt["Sigma_ref"].to(device), img_size


@torch.no_grad()
def score_paths(paths: list[str], ckpt_path: Path, device: torch.device, batch_size: int, workers: int) -> list[float]:
    model, mu_ref, sigma_ref, img_size = load_model(ckpt_path, device)
    loader = DataLoader(
        ImagePathDataset(paths, img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda" and workers > 0),
    )
    scores: list[float] = []
    for imgs, _idx in loader:
        imgs = imgs.to(device, non_blocking=(device.type == "cuda"))
        _, mu, logvar = model(imgs)
        sz = sz_from_stats(mu, logvar, mu_ref, sigma_ref, sigma_t_max=1.0, mu_only=True)
        scores.extend(sz.detach().cpu().numpy().tolist())
    return scores


def compute_v2_holdout(v2_epoch: int, v2_ckpt: Path, device: torch.device, batch_size: int, workers: int) -> pd.DataFrame:
    if OUT_V2_HOLDOUT.exists():
        cached = pd.read_csv(OUT_V2_HOLDOUT)
        if int(cached.attrs.get("epoch", v2_epoch)) == v2_epoch or "E_A_v2" in cached.columns:
            return cached

    day5 = pd.read_csv(DAY5_HOLDOUT)
    kadid = day5[day5["dataset"].eq(KADID)].copy()
    kadid["distorted_path"] = kadid["distorted_path"].astype(str).map(remap_path)
    missing = [p for p in kadid["distorted_path"].tolist() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} KADID holdout image paths. First missing: {missing[0]}")
    paths = kadid["distorted_path"].astype(str).tolist()
    scores = score_paths(paths, v2_ckpt, device, batch_size=batch_size, workers=workers)
    out = kadid[
        [
            "dataset",
            "image_id",
            "ref_id",
            "distorted_path",
            "distortion_type",
            "severity_or_level",
            "mos_or_dmos",
        ]
    ].copy()
    out["v2_epoch"] = int(v2_epoch)
    out["v2_ckpt_path"] = str(v2_ckpt)
    out["E_A_v2"] = scores
    out.to_csv(OUT_V2_HOLDOUT, index=False)
    return out


def val_stats_for_epoch(v2_epoch: int) -> dict[str, float]:
    taskd_stats = pd.read_csv(TASKD_VAL_STATS)
    kadid_h = taskd_stats[taskd_stats["dataset"].eq(KADID)].iloc[0]

    val_scores = pd.read_csv(TASKC_VAL_SCORES)
    val = val_scores[(val_scores["dataset"].eq(KADID)) & (val_scores["epoch"].eq(v2_epoch))]
    if val.empty:
        raise ValueError(f"No KADID val scores found for epoch {v2_epoch} in {TASKC_VAL_SCORES}")
    return {
        "mean_H": float(kadid_h["mean_H"]),
        "std_H": float(kadid_h["std_H"]),
        "mean_A_v2": float(val["E_A"].mean()),
        "std_A_v2": float(val["E_A"].std(ddof=0)),
        "N_val_A_v2": int(len(val)),
    }


def make_compare_scores(v2_epoch: int, v2_holdout: pd.DataFrame) -> pd.DataFrame:
    taskd = pd.read_csv(TASKD_HOLDOUT)
    v1 = taskd[taskd["dataset"].eq(KADID)].copy()
    merged = v1.merge(
        v2_holdout[["image_id", "v2_epoch", "v2_ckpt_path", "E_A_v2"]],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(v1):
        raise ValueError(f"Expected {len(v1)} merged KADID rows, got {len(merged)}")

    stats = val_stats_for_epoch(v2_epoch)
    merged["z_A_v2"] = (merged["E_A_v2"] - stats["mean_A_v2"]) / stats["std_A_v2"]
    merged["Qz_AmH_v2"] = merged["z_A_v2"] - merged["z_H"]
    merged["Qz_HmA_v2"] = merged["z_H"] - merged["z_A_v2"]
    merged = merged.rename(
        columns={
            "E_A": "E_A_epoch5_v1",
            "Qz_AmH": "Qz_AmH_epoch5_v1",
            "Qz_HmA": "Qz_HmA_epoch5_v1",
            "z_A": "z_A_epoch5_v1",
        }
    )
    cols = [
        "dataset",
        "image_id",
        "ref_id",
        "distortion_type",
        "severity_or_level",
        "mos_or_dmos",
        "E_H",
        "E_A_epoch5_v1",
        "E_A_v2",
        "z_H",
        "z_A_epoch5_v1",
        "z_A_v2",
        "Qz_AmH_epoch5_v1",
        "Qz_AmH_v2",
        "Qz_HmA_epoch5_v1",
        "Qz_HmA_v2",
        "v2_epoch",
        "v2_ckpt_path",
    ]
    merged[cols].to_csv(OUT_COMPARE_SCORES, index=False)
    return merged[cols]


def metric_rows(compare: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("SRCC E_A vs MOS", "E_A_epoch5_v1", "E_A_v2", "mos_or_dmos", None),
        ("SRCC Q_z^(A-H) vs MOS", "Qz_AmH_epoch5_v1", "Qz_AmH_v2", "mos_or_dmos", None),
        ("Blur E_A--sev", "E_A_epoch5_v1", "E_A_v2", "severity_or_level", "Gaussian blur"),
        ("Sharpen E_A--sev", "E_A_epoch5_v1", "E_A_v2", "severity_or_level", "High sharpen"),
        ("Pixelate E_A--sev", "E_A_epoch5_v1", "E_A_v2", "severity_or_level", "Pixelate"),
    ]
    rows = []
    for metric, v1_col, v2_col, target, distortion_type in specs:
        df = compare
        if distortion_type is not None:
            df = compare[compare["distortion_type"].eq(distortion_type)]
        v1 = finite_corr(df, v1_col, target)
        v2 = finite_corr(df, v2_col, target)
        better = "v2" if abs(float(v2["SRCC"])) > abs(float(v1["SRCC"])) else "epoch 5 (v1)"
        rows.append(
            {
                "Metric": metric,
                "epoch 5 (v1)": v1["SRCC"],
                "v2 selected": v2["SRCC"],
                "N": v2["N"],
                "Better?": better,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_COMPARE_TABLE, index=False)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection = recompute_selection_v2()
    selected = selection[selection["selected_v2"].eq("yes")].iloc[0]
    v2_epoch = int(selected["Epoch"])
    v2_ckpt = Path(str(selected["ckpt path"]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    workers = 4 if device.type == "cuda" else 0
    batch_size = 64 if device.type == "cuda" else 16

    v2_holdout = compute_v2_holdout(v2_epoch, v2_ckpt, device, batch_size=batch_size, workers=workers)
    compare_scores = make_compare_scores(v2_epoch, v2_holdout)
    compare_table = metric_rows(compare_scores)

    summary = {
        "task": "Day 8 quick check for selection criteria",
        "output_dir": str(OUT_DIR),
        "device_used": str(device),
        "v1_epoch": V1_EPOCH,
        "v2_selected_epoch": v2_epoch,
        "v2_selected_checkpoint": str(v2_ckpt),
        "selection_score_v2_formula": "(-SRCC_EA_sev_blur)+(-SRCC_EA_sev_lens)-max(0,SRCC_EA_sev_sharpen)-max(0,SRCC_EA_sev_pixelate)",
        "kadid_holdout_rows_compared": int(len(compare_scores)),
        "outputs": {
            "selection_table": str(OUT_SELECTION),
            "v2_holdout_scores": str(OUT_V2_HOLDOUT),
            "compare_scores": str(OUT_COMPARE_SCORES),
            "compare_table": str(OUT_COMPARE_TABLE),
            "report": str(OUT_REPORT),
            "summary": str(OUT_SUMMARY),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    report = [
        "Day 8 quick check for selection criteria",
        "",
        f"V1 selected epoch: {V1_EPOCH}",
        f"V2 selected epoch: {v2_epoch}",
        f"V2 selected checkpoint: {v2_ckpt}",
        "",
        "selection_score_v2 = (-blur) + (-lens) - max(0, sharpen) - max(0, pixelate)",
        "",
        "Task C v2 checkpoint selection table:",
        selection.to_string(index=False),
        "",
        "KADID holdout-only epoch 5 vs v2 comparison:",
        compare_table.to_string(index=False),
        "",
        "Notes:",
        "No retraining was done.",
        "The original Day 7 Task B/Task C/Task D files were not deleted or modified.",
        "Epoch 5 values are from Day 7 Task D; v2 selected values are scored on KADID holdout only using the same preprocess/ref setup.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(report))

    print(f"Day 8 complete: {OUT_DIR}")
    print(f"V2 selected epoch: {v2_epoch}")
    print(compare_table.to_string(index=False))


if __name__ == "__main__":
    main()
