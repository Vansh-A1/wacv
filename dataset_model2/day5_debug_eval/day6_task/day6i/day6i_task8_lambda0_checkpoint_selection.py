#!/usr/bin/env python3
"""Day 6 Task 8: select lambda0 checkpoint using validation-only correlations."""
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
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PACKAGE_ROOT = Path("/home/projectwork/student_package")
DATASET_ROOT = PACKAGE_ROOT / "dataset_model2"
DAY5_DIR = DATASET_ROOT / "day5_debug_eval"
OUT_DIR = DAY5_DIR / "day6i"
PLOTS_DIR = OUT_DIR / "plots"

CKPT_DIR = DATASET_ROOT / "checkpoints" / "wacv_degradation_from_scratch_lambda0_seed42"
RUN_REF_STATS = DATASET_ROOT / "runs" / "wacv_degradation_from_scratch_lambda0_seed42" / "reference_stats.pt"
SPLIT_CSV = DATASET_ROOT / "ref_splits_seed42" / "combined_kadid_tid_koniq_split_seed42.csv"
DAY5_HOLDOUT = DAY5_DIR / "day5_holdout_scores.csv"

OUT_VAL_SCORES = OUT_DIR / "task8_val_scores_all_saved_checkpoints.csv"
OUT_VAL_CORR = OUT_DIR / "task8_val_checkpoint_correlations.csv"
OUT_SELECTION = OUT_DIR / "task8_checkpoint_selection.csv"
OUT_HOLDOUT = OUT_DIR / "task8_selected_checkpoint_holdout_scores.csv"
OUT_HOLDOUT_CORR = OUT_DIR / "task8_selected_checkpoint_holdout_correlations.csv"
OUT_JSON = OUT_DIR / "task8_lambda0_checkpoint_selection_summary.json"
OUT_TXT = OUT_DIR / "task8_lambda0_checkpoint_selection_report.txt"

PATH_REMAPS = {
    "/data/projectwork/swati_mam/kadid10k": "/data/projectwork/swati_mam/new model data/kadid10k",
}
DISTORTION_GROUPS = {
    "blur": ["blur"],
    "sharpen": ["sharpen"],
    "pixelate": ["jpeg", "jpeg2000", "color quantization", "color block", "pixel"],
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
    def __init__(self, paths: list[str], img_size: int = 256):
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


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    latent_dim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=latent_dim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    ref_source = "checkpoint"
    if "mu_ref" in ckpt and "Sigma_ref" in ckpt:
        mu_ref = ckpt["mu_ref"]
        sig_ref = ckpt["Sigma_ref"]
    else:
        ref = torch.load(RUN_REF_STATS, map_location=device, weights_only=False)
        mu_ref = ref["mu_ref"]
        sig_ref = ref["Sigma_ref"]
        ref_source = str(RUN_REF_STATS)
    ckpt["_reference_source_used"] = ref_source
    return model, mu_ref.to(device), sig_ref.to(device), img_size, ckpt


@torch.no_grad()
def score_paths(paths: list[str], ckpt_path: Path, device: torch.device, batch_size: int = 64) -> tuple[list[float], dict]:
    model, mu_ref, sig_ref, img_size, ckpt = load_model(ckpt_path, device)
    loader = DataLoader(
        ImagePathDataset(paths, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    scores: list[float] = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        _, mu, logvar = model(imgs)
        sz = sz_from_stats(mu, logvar, mu_ref, sig_ref, sigma_t_max=1.0, mu_only=True)
        scores.extend(sz.detach().cpu().numpy().tolist())
    meta = {
        "checkpoint": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "best_metric": ckpt.get("best_metric"),
        "best_value": float(ckpt.get("best_value")) if ckpt.get("best_value") is not None else None,
        "reference_source_used": ckpt.get("_reference_source_used", "checkpoint"),
    }
    return scores, meta


def corr(x, y) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return np.nan, np.nan, int(mask.sum())
    sp = spearmanr(x[mask], y[mask])
    return float(sp.statistic), float(sp.pvalue), int(mask.sum())


def pearson(x, y) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return np.nan, np.nan, int(mask.sum())
    pr = pearsonr(x[mask], y[mask])
    return float(pr.statistic), float(pr.pvalue), int(mask.sum())


def distortion_group(name: str) -> str | None:
    low = str(name).lower()
    for group, keys in DISTORTION_GROUPS.items():
        if any(k in low for k in keys):
            return group
    return None


def checkpoint_label(path: Path, meta: dict) -> str:
    epoch = meta.get("epoch", -1)
    if epoch and epoch > 0:
        return f"{path.stem}_epoch_{epoch:04d}"
    m = re.search(r"(\d+)", path.stem)
    return f"{path.stem}_{m.group(1)}" if m else path.stem


def score_validation_checkpoints(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    split = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    split["distorted_path"] = split["distorted_path"].map(remap_path)
    val = split[(split["split"] == "val") & (split["dataset"].isin(["KADID-10k", "TID2013"]))].copy()
    val = val[val["mos_or_dmos"].astype(str).str.upper() != "NA"].copy()
    val["mos_or_dmos"] = pd.to_numeric(val["mos_or_dmos"], errors="coerce")
    val["severity_or_level"] = pd.to_numeric(val["severity_or_level"], errors="coerce")
    val["distortion_group"] = val["distortion_type"].map(distortion_group)

    ckpts = sorted(CKPT_DIR.glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found under {CKPT_DIR}")

    all_scores = []
    corr_rows = []
    metas = []
    paths = val["distorted_path"].astype(str).tolist()

    for ckpt_path in ckpts:
        scores, meta = score_paths(paths, ckpt_path, device)
        label = checkpoint_label(ckpt_path, meta)
        meta["label"] = label
        metas.append(meta)
        cur = val.copy()
        cur["checkpoint_label"] = label
        cur["checkpoint_path"] = str(ckpt_path)
        cur["E_A_ckpt"] = scores
        all_scores.append(cur)

        pooled = cur[cur["dataset"].isin(["KADID-10k", "TID2013"])]
        rho, p, n = corr(pooled["E_A_ckpt"], pooled["mos_or_dmos"])
        corr_rows.append({
            "checkpoint_label": label,
            "checkpoint_path": str(ckpt_path),
            "epoch": meta["epoch"],
            "metric": "pooled Spearman(E_A, MOS)",
            "subset": "KADID+TID val with MOS",
            "desired_sign": "negative for higher=worse E_A vs higher=better MOS",
            "spearman_rho": rho,
            "spearman_p": p,
            "N": n,
        })

        kadid = cur[cur["dataset"] == "KADID-10k"]
        for group in ["blur", "sharpen", "pixelate"]:
            sub = kadid[kadid["distortion_group"] == group]
            rho, p, n = corr(sub["E_A_ckpt"], sub["severity_or_level"])
            corr_rows.append({
                "checkpoint_label": label,
                "checkpoint_path": str(ckpt_path),
                "epoch": meta["epoch"],
                "metric": f"within-type Spearman(E_A, severity) for {group}",
                "subset": f"KADID val {group}",
                "desired_sign": "negative if severity increase lowers E_A",
                "spearman_rho": rho,
                "spearman_p": p,
                "N": n,
            })

    return pd.concat(all_scores, ignore_index=True), pd.DataFrame(corr_rows), metas


def choose_checkpoint(corr_df: pd.DataFrame) -> pd.DataFrame:
    wide_rows = []
    for label, g in corr_df.groupby("checkpoint_label"):
        row = {
            "checkpoint_label": label,
            "checkpoint_path": g["checkpoint_path"].iloc[0],
            "epoch": int(g["epoch"].iloc[0]),
        }
        for _, r in g.iterrows():
            row[r["metric"]] = r["spearman_rho"]
        blur = row.get("within-type Spearman(E_A, severity) for blur", np.nan)
        sharpen = row.get("within-type Spearman(E_A, severity) for sharpen", np.nan)
        pixelate = row.get("within-type Spearman(E_A, severity) for pixelate", np.nan)
        pooled = row.get("pooled Spearman(E_A, MOS)", np.nan)
        row["selection_score"] = np.nanmean([
            abs(blur) if np.isfinite(blur) and blur < 0 else -1.0,
            abs(sharpen) if np.isfinite(sharpen) and sharpen < 0 else -1.0,
            abs(pixelate) if np.isfinite(pixelate) and pixelate < 0 else -1.0,
            abs(pooled) if np.isfinite(pooled) else 0.0,
        ])
        row["selection_rule"] = (
            "Maximize negative within-type severity correlations for blur/sharpen/pixelate, "
            "with pooled E_A-vs-MOS as tie/support. No MOS used in VAE loss."
        )
        wide_rows.append(row)
    sel = pd.DataFrame(wide_rows).sort_values("selection_score", ascending=False).reset_index(drop=True)
    sel["selected"] = ["yes" if i == 0 else "no" for i in range(len(sel))]
    return sel


def score_selected_holdout(selected_path: Path, selected_label: str, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    hold = pd.read_csv(DAY5_HOLDOUT)
    hold = hold[(hold["dataset"].isin(["KADID-10k", "TID2013"])) & (hold["has_mos"] == True)].copy()
    paths = hold["distorted_path"].astype(str).map(remap_path).tolist()
    scores, _ = score_paths(paths, selected_path, device)
    hold["selected_checkpoint_label"] = selected_label
    hold["E_A_selected"] = scores
    hold.to_csv(OUT_HOLDOUT, index=False)

    rows = []
    for label, sub in [
        ("KADID holdout", hold[hold["dataset"] == "KADID-10k"]),
        ("TID holdout", hold[hold["dataset"] == "TID2013"]),
        ("pooled KADID+TID holdout", hold),
    ]:
        sr, sp, n = corr(sub["E_A_selected"], sub["mos_or_dmos"])
        pr, pp, _ = pearson(sub["E_A_selected"], sub["mos_or_dmos"])
        rows.append({
            "checkpoint_label": selected_label,
            "subset": label,
            "N": n,
            "spearman_E_A_selected_vs_MOS": sr,
            "spearman_p": sp,
            "pearson_E_A_selected_vs_MOS": pr,
            "pearson_p": pp,
        })
    hcorr = pd.DataFrame(rows)
    hcorr.to_csv(OUT_HOLDOUT_CORR, index=False)
    return hold, hcorr


def make_plots(corr_df: pd.DataFrame, hcorr: pd.DataFrame) -> list[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    pivot = corr_df.pivot(index="checkpoint_label", columns="metric", values="spearman_rho")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Validation Spearman rho")
    ax.set_title("Task 8 validation-only checkpoint selection metrics")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p = PLOTS_DIR / "task8_validation_checkpoint_correlations.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(hcorr["subset"], hcorr["spearman_E_A_selected_vs_MOS"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Holdout Spearman rho")
    ax.set_title("Selected checkpoint on holdout")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    p = PLOTS_DIR / "task8_selected_checkpoint_holdout_correlations.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(str(p))
    return made


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_scores, val_corr, metas = score_validation_checkpoints(device)
    val_scores.to_csv(OUT_VAL_SCORES, index=False)
    val_corr.to_csv(OUT_VAL_CORR, index=False)

    selection = choose_checkpoint(val_corr)
    selection.to_csv(OUT_SELECTION, index=False)
    selected = selection.iloc[0]
    holdout, hcorr = score_selected_holdout(
        Path(selected["checkpoint_path"]),
        str(selected["checkpoint_label"]),
        device,
    )
    plots = make_plots(val_corr, hcorr)

    summary = {
        "task": "Task 8: lambda0 checkpoint selection using validation only, then one holdout evaluation",
        "available_checkpoints": metas,
        "note": "Only best.pth and last.pth were present; no separate epoch 25/35/45 checkpoint files were found.",
        "selection_rule": str(selected["selection_rule"]),
        "selected_checkpoint": selected.to_dict(),
        "val_rows_scored": int(len(val_scores)),
        "holdout_rows_scored": int(len(holdout)),
        "holdout_correlations": hcorr.to_dict("records"),
        "outputs": {
            "val_scores": str(OUT_VAL_SCORES),
            "val_correlations": str(OUT_VAL_CORR),
            "selection": str(OUT_SELECTION),
            "selected_holdout_scores": str(OUT_HOLDOUT),
            "selected_holdout_correlations": str(OUT_HOLDOUT_CORR),
            "summary_json": str(OUT_JSON),
            "report_txt": str(OUT_TXT),
            "plots": plots,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    report = [
        "Day 6 Task 8: lambda0 checkpoint selection",
        "",
        f"Checkpoint directory: {CKPT_DIR}",
        "Found checkpoints:",
    ]
    report.extend(f"- {m['label']}: {m['checkpoint']} (epoch={m['epoch']})" for m in metas)
    report.extend([
        "",
        "Validation-only metrics:",
        val_corr.to_string(index=False),
        "",
        "Selection table:",
        selection.to_string(index=False),
        "",
        f"Selected checkpoint: {selected['checkpoint_label']}",
        f"Selected path: {selected['checkpoint_path']}",
        "",
        "Holdout evaluation of selected checkpoint:",
        hcorr.to_string(index=False),
        "",
        "Important note:",
        "- Selection used validation correlations only.",
        "- MOS was not used inside the VAE loss.",
        "- Only the selected checkpoint was evaluated on holdout in this task.",
        "- No epoch 25/35/45 checkpoint files were present; best.pth and last.pth were evaluated.",
        "",
        "Outputs:",
        f"- {OUT_VAL_SCORES}",
        f"- {OUT_VAL_CORR}",
        f"- {OUT_SELECTION}",
        f"- {OUT_HOLDOUT}",
        f"- {OUT_HOLDOUT_CORR}",
        f"- {OUT_JSON}",
        f"- {OUT_TXT}",
        f"- {PLOTS_DIR}",
    ])
    OUT_TXT.write_text("\n".join(report) + "\n")
    print(f"Saved Task 8 outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
