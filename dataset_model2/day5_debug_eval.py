#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from torchvision import transforms


PACKAGE_ROOT = Path("/home/projectwork/student_package")
DATASET_ROOT = PACKAGE_ROOT / "dataset_model2"
OUT_DIR = DATASET_ROOT / "day5_debug_eval"
SPLIT_CSV = DATASET_ROOT / "ref_splits_seed42" / "combined_kadid_tid_koniq_split_seed42.csv"
TRAIN_CSV = DATASET_ROOT / "training_manifests" / "degradation_train_seed42.csv"
VAL_EVAL_LIST = DATASET_ROOT / "runs" / "wacv_degradation_from_scratch_lambda0_seed42" / "eval_files.txt"

EH_CKPT = PACKAGE_ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"
EA_CKPT = DATASET_ROOT / "checkpoints" / "wacv_degradation_from_scratch_lambda0_seed42" / "best.pth"

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
            return new + path[len(old) :]
    return path


def tensor_stats(t: torch.Tensor) -> dict:
    td = t.detach().float().cpu()
    return {
        "shape": list(td.shape),
        "finite": bool(torch.isfinite(td).all().item()),
        "nan_count": int(torch.isnan(td).sum().item()),
        "inf_count": int(torch.isinf(td).sum().item()),
        "min": float(td.min().item()) if td.numel() else float("nan"),
        "max": float(td.max().item()) if td.numel() else float("nan"),
        "mean": float(td.mean().item()) if td.numel() else float("nan"),
        "std": float(td.std(unbiased=False).item()) if td.numel() else float("nan"),
    }


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    img_size = int(cfg.get("img") or cfg.get("image_size") or 256)
    ldim = int(cfg.get("ldim") or 100)
    model = CVAEGenerator_v2(latent_dim=ldim, image_size=img_size).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    mu_ref = ckpt["mu_ref"].to(device)
    sigma_ref = ckpt["Sigma_ref"].to(device)
    cfg["sz_mode"] = ckpt.get("sz_mode") or cfg.get("sz_mode") or "mu_only"
    cfg["sz_sigma_t_max"] = float(
        ckpt.get("sz_sigma_t_max")
        if ckpt.get("sz_sigma_t_max") is not None
        else cfg.get("sz_sigma_t_max", 1.0)
    )
    return model, mu_ref, sigma_ref, img_size, ldim, cfg, ckpt


def checkpoint_diagnostics(name: str, ckpt_path: Path, ckpt: dict, mu_ref: torch.Tensor, sigma_ref: torch.Tensor) -> dict:
    bad_params = []
    for key, val in ckpt.get("generator", {}).items():
        if torch.is_tensor(val) and not torch.isfinite(val).all():
            bad_params.append(key)
    return {
        "name": name,
        "checkpoint": str(ckpt_path),
        "exists": ckpt_path.is_file(),
        "epoch": ckpt.get("epoch"),
        "best_metric": ckpt.get("best_metric"),
        "best_value": ckpt.get("best_value"),
        "mu_ref": tensor_stats(mu_ref),
        "Sigma_ref": tensor_stats(sigma_ref),
        "bad_parameter_count": len(bad_params),
        "bad_parameters": bad_params[:20],
    }


def load_image_tensor(path: str, img_size: int, device: torch.device):
    pil = Image.open(path).convert("RGB")
    original_size = list(pil.size)
    pil = _resize_short_side(pil, img_size)
    pil = center_crop(pil, img_size)
    arr = (np.asarray(pil) / 255.0).astype("float32")
    tensor = transforms.ToTensor()(arr).unsqueeze(0).to(device)
    return tensor, original_size, list(pil.size)


@torch.no_grad()
def score_one(model, mu_ref, sigma_ref, cfg, x: torch.Tensor):
    recon, mu, log_var = model(x)
    sigma_t = torch.exp(log_var)
    if cfg.get("sz_sigma_t_max") is not None and float(cfg["sz_sigma_t_max"]) > 0:
        sigma_t = torch.clamp(sigma_t, max=float(cfg["sz_sigma_t_max"]))
    den = sigma_ref.to(x.device).view(1, -1) + 1e-8
    if cfg.get("sz_mode", "mu_only") != "mu_only":
        den = den + sigma_t
    num = (mu_ref.to(x.device).view(1, -1) - mu) ** 2
    frac = num / den
    sz_manual = torch.sqrt(torch.sum(frac, dim=1))
    sz_func = sz_from_stats(
        mu,
        log_var,
        mu_ref,
        sigma_ref,
        sigma_t_max=cfg.get("sz_sigma_t_max", 1.0),
        mu_only=(cfg.get("sz_mode", "mu_only") == "mu_only"),
    )
    return {
        "score": float(sz_func.view(-1)[0].detach().cpu().item()),
        "manual_score": float(sz_manual.view(-1)[0].detach().cpu().item()),
        "input": tensor_stats(x),
        "recon": tensor_stats(recon),
        "mu": tensor_stats(mu),
        "log_var": tensor_stats(log_var),
        "sigma_t": tensor_stats(sigma_t),
        "den": tensor_stats(den),
        "num": tensor_stats(num),
        "num_over_den": tensor_stats(frac),
        "score_tensor": tensor_stats(sz_func),
    }


def reconstruct_reference_entries(max_n: int = 2000, seed: int = 123) -> list[tuple[str, str]]:
    df = pd.read_csv(TRAIN_CSV, dtype=str, keep_default_na=False)
    entries = [(p, pool) for p, pool in zip(df["path"], df["pool"])]
    if len(entries) <= max_n:
        return entries
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(entries), size=max_n, replace=False)
    return [entries[i] for i in sorted(idx.tolist())]


def diagnose_reference_set(model, mu_ref, sigma_ref, cfg, img_size, device, limit=None):
    rows = []
    entries = reconstruct_reference_entries()
    if limit is not None:
        entries = entries[:limit]
    for i, (path, pool) in enumerate(entries):
        path = remap_path(path)
        row = {"index": i, "path": path, "pool": pool, "loaded": False, "finite_score": False}
        try:
            x, original_size, crop_size = load_image_tensor(path, img_size, device)
            out = score_one(model, mu_ref, sigma_ref, cfg, x)
            row.update(
                {
                    "loaded": True,
                    "original_size": original_size,
                    "crop_size": crop_size,
                    "input_finite": out["input"]["finite"],
                    "mu_finite": out["mu"]["finite"],
                    "logvar_finite": out["log_var"]["finite"],
                    "den_min": out["den"]["min"],
                    "score": out["score"],
                    "finite_score": math.isfinite(out["score"]),
                }
            )
        except Exception as exc:
            row["error"] = repr(exc)
        rows.append(row)
    return rows


def read_eval_list_label_counts() -> dict:
    def label_from_basename(path: str) -> str:
        b = os.path.basename(path).lower()
        if "hr" in b:
            return "HR"
        if "lr" in b:
            return "LR"
        if "gblur" in b or "blur" in b:
            return "gblur"
        if "jpeg" in b:
            return "jpeg"
        return "other"

    paths = [ln.strip() for ln in VAL_EVAL_LIST.read_text().splitlines() if ln.strip()]
    counts = {}
    for p in paths:
        lab = label_from_basename(p)
        counts[lab] = counts.get(lab, 0) + 1
    return {"eval_list": str(VAL_EVAL_LIST), "n": len(paths), "label_counts": counts}


def load_holdout_manifest() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_CSV, dtype=str, keep_default_na=False)
    df = df[df["split"] == "holdout"].copy()
    df["distorted_path"] = df["distorted_path"].map(remap_path)
    df["ref_path"] = df["ref_path"].map(lambda p: remap_path(p) if p != "NA" else p)
    df["has_mos"] = df["mos_or_dmos"] != "NA"
    df["path_exists"] = df["distorted_path"].map(lambda p: Path(p).is_file())
    return df


def score_manifest(df, eh_bundle, ea_bundle, device, max_rows=None) -> pd.DataFrame:
    eh_model, eh_mu, eh_sig, eh_img, _eh_ldim, eh_cfg = eh_bundle
    ea_model, ea_mu, ea_sig, ea_img, _ea_ldim, ea_cfg = ea_bundle
    rows = []
    work = df[df["path_exists"]].copy()
    if max_rows is not None:
        work = work.head(max_rows)
    for idx, row in work.iterrows():
        out = row.to_dict()
        try:
            x_h, original_size, crop_size_h = load_image_tensor(row["distorted_path"], eh_img, device)
            x_a, _original_size_a, crop_size_a = load_image_tensor(row["distorted_path"], ea_img, device)
            sh = score_one(eh_model, eh_mu, eh_sig, eh_cfg, x_h)
            sa = score_one(ea_model, ea_mu, ea_sig, ea_cfg, x_a)
            e_h = sh["score"]
            e_a = sa["score"]
            out.update(
                {
                    "original_size": original_size,
                    "EH_crop_size": crop_size_h,
                    "EA_crop_size": crop_size_a,
                    "E_H": e_h,
                    "E_A": e_a,
                    "Q": e_a - e_h,
                    "E_H_finite": math.isfinite(e_h),
                    "E_A_finite": math.isfinite(e_a),
                    "Q_finite": math.isfinite(e_a - e_h),
                    "input_finite_EH": sh["input"]["finite"],
                    "mu_finite_EH": sh["mu"]["finite"],
                    "logvar_finite_EH": sh["log_var"]["finite"],
                    "den_min_EH": sh["den"]["min"],
                    "input_finite_EA": sa["input"]["finite"],
                    "mu_finite_EA": sa["mu"]["finite"],
                    "logvar_finite_EA": sa["log_var"]["finite"],
                    "den_min_EA": sa["den"]["min"],
                }
            )
        except Exception as exc:
            out.update({"error": repr(exc)})
        rows.append(out)
    return pd.DataFrame(rows)


def corr_pair(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return {"N": n, "spearman_rho": np.nan, "spearman_p": np.nan, "pearson_r": np.nan, "pearson_p": np.nan, "kendall_tau": np.nan, "kendall_p": np.nan}
    sp = spearmanr(x[mask], y[mask])
    pr = pearsonr(x[mask], y[mask])
    kt = kendalltau(x[mask], y[mask])
    return {
        "N": n,
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "kendall_tau": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
    }


def compute_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mos_df = scores[(scores["has_mos"] == True) & scores["mos_or_dmos"].notna()].copy()
    mos_df["mos_float"] = pd.to_numeric(mos_df["mos_or_dmos"], errors="coerce")
    for dataset_name, subset in [("pooled_KADID_TID", mos_df)] + list(mos_df.groupby("dataset")):
        for target_name, target in [("MOS_or_DMOS", subset["mos_float"]), ("negative_MOS_or_DMOS", -subset["mos_float"])]:
            for score_name in ["E_H", "E_A", "Q"]:
                c = corr_pair(subset[score_name], target)
                rows.append({"dataset": dataset_name, "target": target_name, "score": score_name, **c})
    return pd.DataFrame(rows)


def plot_scatters(scores: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    df = scores[(scores["has_mos"] == True)].copy()
    df["mos_float"] = pd.to_numeric(df["mos_or_dmos"], errors="coerce")
    for dataset_name, subset in [("pooled_KADID_TID", df)] + list(df.groupby("dataset")):
        for score_name in ["E_H", "E_A", "Q"]:
            valid = subset[[score_name, "mos_float"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 3:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(valid[score_name], valid["mos_float"], s=10, alpha=0.55)
            ax.set_xlabel(score_name)
            ax.set_ylabel("MOS/DMOS")
            ax.set_title(f"{dataset_name}: {score_name} vs MOS/DMOS")
            fig.tight_layout()
            path = out_dir / f"{dataset_name}_{score_name}_vs_mos.png"
            fig.savefig(path, dpi=140)
            plt.close(fig)
            made.append(str(path))
    return made


def distortion_bar(scores: pd.DataFrame, out_dir: Path) -> str | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    k = scores[(scores["dataset"] == "KADID-10k") & scores["Q_finite"]].copy()
    if len(k) == 0:
        return None
    grouped = k.groupby("distortion_type")["Q"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(grouped))))
    grouped.plot(kind="barh", ax=ax)
    ax.set_xlabel("mean Q = E_A - E_H")
    ax.set_title("KADID holdout distortion-type mean Q")
    fig.tight_layout()
    path = out_dir / "kadid_distortion_type_mean_Q.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    eh_model, eh_mu, eh_sig, eh_img, eh_ldim, eh_cfg, eh_ckpt = load_model(EH_CKPT, device)
    ea_model, ea_mu, ea_sig, ea_img, ea_ldim, ea_cfg, ea_ckpt = load_model(EA_CKPT, device)

    diag = {
        "device": str(device),
        "requested_missing_checkpoint_note": {
            "pasted_request_checkpoint": str(DATASET_ROOT / "checkpoints" / "wacv_degradation_from_scratch_seed42" / "best.pth"),
            "exists": (DATASET_ROOT / "checkpoints" / "wacv_degradation_from_scratch_seed42" / "best.pth").is_file(),
            "used_degradation_checkpoint": str(EA_CKPT),
        },
        "checkpoint_diagnostics": [
            checkpoint_diagnostics("E_H_pristine_model", EH_CKPT, eh_ckpt, eh_mu, eh_sig),
            checkpoint_diagnostics("E_A_degradation_model", EA_CKPT, ea_ckpt, ea_mu, ea_sig),
        ],
        "validation_nan_trace": read_eval_list_label_counts(),
        "preprocessing": {
            "training_degradation": "PIL RGB -> resize short side to >=256 if needed -> random 256x256 crop -> random hflip -> float RGB [0,1] -> ToTensor; no mean/std normalization",
            "validation_degradation": "PIL RGB -> resize short side to >=256 if needed -> center 256x256 crop -> float RGB [0,1] -> ToTensor; no mean/std normalization",
            "day5_EH_EA": "Both E_H and E_A use PIL RGB -> resize short side to >=256 if needed -> center 256x256 crop -> float RGB [0,1] -> ToTensor; no mean/std normalization",
        },
        "sz_math": {
            "mode": "mu_only",
            "formula": "sqrt(sum((mu_ref - mu_t)^2 / (Sigma_ref + eps)))",
            "eps": 1e-8,
            "Sigma_ref_clamp_in_fit_reference": "clamp_min(1e-8)",
        },
    }

    ref_rows_full = diagnose_reference_set(ea_model, ea_mu, ea_sig, ea_cfg, ea_img, device)
    ref_df = pd.DataFrame(ref_rows_full)
    ref_df.to_csv(OUT_DIR / "ea_reference_set_diagnostics.csv", index=False)
    diag["ea_reference_set"] = {
        "attempted": int(len(ref_df)),
        "loaded": int(ref_df.get("loaded", pd.Series(dtype=bool)).sum()),
        "skipped": int((~ref_df.get("loaded", pd.Series(dtype=bool))).sum()),
        "finite_score": int(ref_df.get("finite_score", pd.Series(dtype=bool)).sum()),
        "nan_or_inf_score": int((ref_df.get("loaded", False) & ~ref_df.get("finite_score", False)).sum()),
        "score_summary": {
            "min": float(ref_df["score"].min()),
            "max": float(ref_df["score"].max()),
            "mean": float(ref_df["score"].mean()),
            "std": float(ref_df["score"].std(ddof=0)),
        },
        "note": "These are the 2000 center-crop entries reconstructed from the degradation training list. The code comment says pristine, but this degradation run used the degradation train_list.",
    }

    holdout = load_holdout_manifest()
    small = pd.concat(
        [
            holdout[(holdout["dataset"] == "KADID-10k") & holdout["path_exists"]].head(10),
            holdout[(holdout["dataset"] == "TID2013") & holdout["path_exists"]].head(10),
        ],
        ignore_index=True,
    )
    small_scores = score_manifest(
        small,
        (eh_model, eh_mu, eh_sig, eh_img, eh_ldim, eh_cfg),
        (ea_model, ea_mu, ea_sig, ea_img, ea_ldim, ea_cfg),
        device,
    )
    small_scores.to_csv(OUT_DIR / "small_diagnostic_scores.csv", index=False)
    diag["small_diagnostic"] = {
        "rows": int(len(small_scores)),
        "E_H_finite": int(small_scores["E_H_finite"].sum()),
        "E_A_finite": int(small_scores["E_A_finite"].sum()),
        "Q_finite": int(small_scores["Q_finite"].sum()),
        "E_H_mean": float(pd.to_numeric(small_scores["E_H"]).mean()),
        "E_A_mean": float(pd.to_numeric(small_scores["E_A"]).mean()),
        "Q_mean": float(pd.to_numeric(small_scores["Q"]).mean()),
    }
    if not (
        small_scores["E_H_finite"].all()
        and small_scores["E_A_finite"].all()
        and small_scores["Q_finite"].all()
    ):
        raise RuntimeError("Small diagnostic found non-finite E_H/E_A/Q; stopping before full evaluation.")

    scores = score_manifest(
        holdout,
        (eh_model, eh_mu, eh_sig, eh_img, eh_ldim, eh_cfg),
        (ea_model, ea_mu, ea_sig, ea_img, ea_ldim, ea_cfg),
        device,
    )
    scores.to_csv(OUT_DIR / "day5_holdout_scores.csv", index=False)

    corr = compute_correlations(scores)
    corr.to_csv(OUT_DIR / "day5_correlations.csv", index=False)
    plots = plot_scatters(scores, OUT_DIR / "plots")
    bar = distortion_bar(scores, OUT_DIR / "plots")
    if bar:
        plots.append(bar)

    finite_mask = scores[["E_H_finite", "E_A_finite", "Q_finite"]].all(axis=1)
    winner_rows = []
    mos_corr = corr[corr["target"] == "MOS_or_DMOS"].copy()
    for dataset_name, subset in mos_corr.groupby("dataset"):
        subset = subset.dropna(subset=["spearman_rho"])
        if len(subset):
            winner = subset.iloc[subset["spearman_rho"].abs().argmax()].to_dict()
            winner_rows.append(winner)
    winners = pd.DataFrame(winner_rows)
    winners.to_csv(OUT_DIR / "day5_winners.csv", index=False)

    pooled = corr[(corr["dataset"] == "pooled_KADID_TID") & (corr["target"] == "MOS_or_DMOS")]
    srcc = {r["score"]: float(r["spearman_rho"]) for _, r in pooled.iterrows()}
    diag["full_holdout"] = {
        "rows_scored": int(len(scores)),
        "path_exists_rows": int(scores["path_exists"].sum()) if "path_exists" in scores else int(len(scores)),
        "rows_with_mos": int((scores["has_mos"] == True).sum()),
        "finite_all": int(finite_mask.sum()),
        "nonfinite_rows": int((~finite_mask).sum()),
        "E_H_summary": scores["E_H"].astype(float).describe().to_dict(),
        "E_A_summary": scores["E_A"].astype(float).describe().to_dict(),
        "Q_summary": scores["Q"].astype(float).describe().to_dict(),
        "pooled_spearman_mos": srcc,
        "delta_Q_minus_EH_abs_srcc": float(abs(srcc.get("Q", np.nan)) - abs(srcc.get("E_H", np.nan))),
        "auroc_note": "Training validation AUROC is undefined when eval list has one class only. For Day 5, no binary HR/degraded labels were created because scoring is MOS/DMOS correlation based.",
    }
    diag["outputs"] = {
        "diagnostics_json": str(OUT_DIR / "diagnostics.json"),
        "debug_report": str(OUT_DIR / "debug_report.txt"),
        "ea_reference_set_diagnostics": str(OUT_DIR / "ea_reference_set_diagnostics.csv"),
        "small_diagnostic_scores": str(OUT_DIR / "small_diagnostic_scores.csv"),
        "day5_holdout_scores": str(OUT_DIR / "day5_holdout_scores.csv"),
        "day5_correlations": str(OUT_DIR / "day5_correlations.csv"),
        "day5_winners": str(OUT_DIR / "day5_winners.csv"),
        "plots": plots,
    }
    (OUT_DIR / "diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")

    root_cause = (
        "HR mean S_Z=nan in train.py validation is caused by an empty HR subset: "
        "the eval list contains only distorted/authentic validation images, whose basenames do not contain the HR label. "
        "run_validation computes sz_hr = sz_np[is_hr == 1], then returns nan when len(sz_hr)==0. "
        "This is not a NaN from input tensors, checkpoint parameters, mu/log_var, Sigma_ref, or S_Z math."
    )
    lines = [
        "Day 5 VAE/IQA NaN Debug Report",
        "",
        f"Root cause: {root_cause}",
        "",
        "Responsible code:",
        "- train.py run_validation(): labels come from _label_from_basename; sz_hr is empty for the current eval list.",
        "- train.py return block: sz_mean_hr is set to float('nan') when len(sz_hr)==0.",
        "",
        "Fix/handling:",
        "- No architecture/checkpoint/training objective was changed.",
        "- Day 5 evaluation now computes E_H and E_A directly per image using the two frozen checkpoints and identical preprocessing.",
        "- No torch.nan_to_num replacement was used.",
        "",
        f"E_A reference diagnostic: attempted={diag['ea_reference_set']['attempted']}, loaded={diag['ea_reference_set']['loaded']}, finite_score={diag['ea_reference_set']['finite_score']}, nan_or_inf_score={diag['ea_reference_set']['nan_or_inf_score']}",
        f"Small diagnostic: rows={diag['small_diagnostic']['rows']}, E_H_finite={diag['small_diagnostic']['E_H_finite']}, E_A_finite={diag['small_diagnostic']['E_A_finite']}, Q_finite={diag['small_diagnostic']['Q_finite']}",
        f"Full holdout: rows={diag['full_holdout']['rows_scored']}, finite_all={diag['full_holdout']['finite_all']}, nonfinite_rows={diag['full_holdout']['nonfinite_rows']}",
        "",
        "Pooled KADID+TID Spearman vs MOS/DMOS:",
    ]
    for score_name, value in srcc.items():
        lines.append(f"- {score_name}: {value:.6f}")
    lines.extend(
        [
            f"delta_Q_minus_EH_abs_srcc: {diag['full_holdout']['delta_Q_minus_EH_abs_srcc']:.6f}",
            "",
            "Generated outputs:",
        ]
    )
    for key, value in diag["outputs"].items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            lines.extend(f"  {v}" for v in value)
        else:
            lines.append(f"- {key}: {value}")
    (OUT_DIR / "debug_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
