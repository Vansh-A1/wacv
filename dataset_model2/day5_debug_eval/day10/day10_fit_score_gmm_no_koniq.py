#!/usr/bin/env python3
"""Day 10: KADID+TID-only K=5 GMM scoring, no KonIQ images."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr
from sklearn.mixture import GaussianMixture


ROOT = Path("/home/projectwork/student_package")
DATASET_ROOT = ROOT / "dataset_model2"
DEBUG_ROOT = DATASET_ROOT / "day5_debug_eval"
OUT_ROOT = DEBUG_ROOT / "day10"
OUT_DIR = OUT_ROOT / "day10_gmm" / "ep05"

DAY9A_EP05 = DEBUG_ROOT / "day9" / "day9_taskA" / "day9_gmm" / "ep05"
DAY9B = DEBUG_ROOT / "day9" / "day9_taskb"
TASKD_HOLDOUT = DEBUG_ROOT / "day7" / "day7_taskD" / "taskD_holdout_scores.csv"
TASKD_VAL_STATS = DEBUG_ROOT / "day7" / "day7_taskD" / "taskD_val_normalization_stats.csv"

TRAIN_MU = DAY9A_EP05 / "mu_train.npy"
TRAIN_IDS = DAY9A_EP05 / "mu_train_ids.csv"
VAL_MU = DAY9B / "encoded_mu_cache" / "ep05_val_kadid_tid_mu.npy"
VAL_IDS = DAY9B / "encoded_mu_cache" / "ep05_val_kadid_tid_ids.csv"
HOLDOUT_MU = DAY9B / "encoded_mu_cache" / "ep05_kadid_holdout_mu.npy"
HOLDOUT_IDS = DAY9B / "encoded_mu_cache" / "ep05_kadid_holdout_ids.csv"
DAY9_HOLDOUT_METRICS = DAY9B / "holdout_selected_soft_metrics.csv"

SEED = 42
K_FREE = 5
MIN_FAMILY_ROWS = 50
DATASETS = ["KADID-10k", "TID2013"]
KADID = "KADID-10k"


def tid_code(distortion_type: str) -> int | None:
    text = str(distortion_type).strip().lower()
    if text.startswith("type_"):
        try:
            return int(text.split("_", 1)[1])
        except ValueError:
            return None
    return None


def family_for_distortion(distortion_type: str) -> str:
    text = str(distortion_type).strip().lower()
    code = tid_code(text)
    if code is not None:
        if code in {8, 23}:
            return "blur"
        if code in {1, 2, 3, 4, 5, 6, 7, 19, 20}:
            return "noise"
        if code in {10, 11, 12, 13, 21}:
            return "compression"
        if code in {16, 17, 18, 22}:
            return "color"
        return "spatial_detail"

    if text in {"gaussian blur", "lens blur", "motion blur"}:
        return "blur"
    if text in {"white noise", "white noise in color component", "impulse noise", "multiplicative noise"}:
        return "noise"
    if text in {"jpeg", "jpeg2000"}:
        return "compression"
    if text in {
        "color diffusion",
        "color shift",
        "color quantization",
        "color saturation 1",
        "color saturation 2",
        "brighten",
        "darken",
        "mean shift",
        "contrast change",
    }:
        return "color"
    return "spatial_detail"


def require_inputs() -> None:
    required = [
        TRAIN_MU,
        TRAIN_IDS,
        VAL_MU,
        VAL_IDS,
        HOLDOUT_MU,
        HOLDOUT_IDS,
        TASKD_HOLDOUT,
        TASKD_VAL_STATS,
        DAY9_HOLDOUT_METRICS,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def save_gmm_dict(gmm: GaussianMixture, path: Path, extra: dict) -> None:
    payload = {
        "model_type": "sklearn.mixture.GaussianMixture",
        "sklearn_version": __import__("sklearn").__version__,
        "covariance_type": "diag",
        "K": int(gmm.n_components),
        "seed": SEED,
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
    payload.update(extra)
    torch.save(payload, path)


def fit_free(mu: np.ndarray) -> tuple[Path, dict]:
    started = time.time()
    gmm = GaussianMixture(
        n_components=K_FREE,
        covariance_type="diag",
        random_state=SEED,
        reg_covar=1e-6,
        max_iter=300,
        n_init=3,
        init_params="kmeans",
    )
    gmm.fit(mu)
    path = OUT_DIR / "gmm_K05_free.pt"
    meta = {
        "arm": "free",
        "train_N": int(len(mu)),
        "latent_dim": int(mu.shape[1]),
        "fit_seconds": float(time.time() - started),
    }
    save_gmm_dict(gmm, path, meta)
    return path, {**meta, "model_path": str(path), "converged": bool(gmm.converged_), "n_iter": int(gmm.n_iter_), "lower_bound": float(gmm.lower_bound_)}


def build_family_map(train_ids: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for distortion_type in sorted(train_ids["distortion_type"].astype(str).unique()):
        rows.append({"distortion_type": distortion_type, "family_id_initial": family_for_distortion(distortion_type)})
    fam = pd.DataFrame(rows)
    counts = train_ids.assign(family_id_initial=train_ids["distortion_type"].map(family_for_distortion)).groupby("family_id_initial").size()
    fam["family_train_N_initial"] = fam["family_id_initial"].map(counts).astype(int)
    fam["family_id_final"] = fam["family_id_initial"]
    fam.loc[fam["family_train_N_initial"] < MIN_FAMILY_ROWS, "family_id_final"] = "other"
    final_counts = train_ids.assign(family_id_final=train_ids["distortion_type"].map(dict(zip(fam["distortion_type"], fam["family_id_final"])))).groupby("family_id_final").size()
    fam["family_train_N_final"] = fam["family_id_final"].map(final_counts).astype(int)
    return fam.sort_values(["family_id_final", "distortion_type"]).reset_index(drop=True)


def fit_family(mu: np.ndarray, train_ids: pd.DataFrame, family_map: pd.DataFrame) -> tuple[Path, dict]:
    started = time.time()
    map_dict = dict(zip(family_map["distortion_type"], family_map["family_id_final"]))
    labels = train_ids["distortion_type"].map(map_dict).astype(str).to_numpy()
    families = sorted(pd.unique(labels))
    if len(families) > K_FREE:
        raise ValueError(f"Family init produced {len(families)} families, expected <= {K_FREE}: {families}")
    means = np.vstack([mu[labels == fam].mean(axis=0) for fam in families])
    weights = np.asarray([(labels == fam).mean() for fam in families], dtype=np.float64)
    gmm = GaussianMixture(
        n_components=len(families),
        covariance_type="diag",
        random_state=SEED,
        reg_covar=1e-6,
        max_iter=300,
        n_init=1,
        init_params="random",
        means_init=means,
        weights_init=weights,
    )
    gmm.fit(mu)
    path = OUT_DIR / f"gmm_K{len(families):02d}_family.pt"
    meta = {
        "arm": "family",
        "train_N": int(len(mu)),
        "latent_dim": int(mu.shape[1]),
        "families": families,
        "min_family_rows": MIN_FAMILY_ROWS,
        "fit_seconds": float(time.time() - started),
    }
    save_gmm_dict(gmm, path, meta)
    return path, {**meta, "model_path": str(path), "K": len(families), "converged": bool(gmm.converged_), "n_iter": int(gmm.n_iter_), "lower_bound": float(gmm.lower_bound_)}


def gmm_energies(mu: np.ndarray, gmm_path: Path) -> tuple[np.ndarray, np.ndarray]:
    gmm = torch.load(gmm_path, map_location="cpu", weights_only=False)
    weights = np.asarray(gmm["weights"], dtype=np.float64)
    means = np.asarray(gmm["means"], dtype=np.float64)
    cov = np.asarray(gmm["covariances"], dtype=np.float64)
    x = np.asarray(mu, dtype=np.float64)
    diff = x[:, None, :] - means[None, :, :]
    log_comp = (
        np.log(weights[None, :])
        - 0.5 * np.sum(np.log(2.0 * np.pi * cov[None, :, :]) + (diff * diff) / cov[None, :, :], axis=2)
    )
    soft = -logsumexp(log_comp, axis=1)
    hard = np.min(-log_comp, axis=1)
    return soft.astype(np.float32), hard.astype(np.float32)


def corr(x, y) -> tuple[float, float, int]:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return math.nan, math.nan, n
    sp = spearmanr(x, y)
    return float(sp.statistic), float(sp.pvalue), n


def pcorr(x, y) -> tuple[float, float, int]:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return math.nan, math.nan, n
    pr = pearsonr(x, y)
    return float(pr.statistic), float(pr.pvalue), n


def add_arm_scores(df: pd.DataFrame, mu: np.ndarray, models: dict[str, Path]) -> pd.DataFrame:
    out = df.copy()
    for arm, path in models.items():
        soft, hard = gmm_energies(mu, path)
        out[f"E_A_{arm}_soft"] = soft
        out[f"E_A_{arm}_hard"] = hard
    return out


def val_metrics(val_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ["free", "family"]:
        for variant in ["soft", "hard"]:
            col = f"E_A_{arm}_{variant}"
            row = {"arm": arm, "variant": variant, "score_col": col}
            for distortion_type, short in [
                ("Gaussian blur", "blur"),
                ("Lens blur", "lens"),
                ("High sharpen", "sharpen"),
                ("Pixelate", "pixelate"),
            ]:
                sub = val_scores[(val_scores["dataset"].eq(KADID)) & (val_scores["distortion_type"].eq(distortion_type))]
                rho, p, n = corr(sub[col], sub["severity_or_level"])
                row[f"{short}_EA_sev_SRCC"] = rho
                row[f"{short}_EA_sev_p"] = p
                row[f"{short}_N"] = n
            pooled = val_scores[val_scores["dataset"].isin(DATASETS)]
            rho, p, n = corr(pooled[col], pooled["mos_or_dmos"])
            row["pooled_EA_MOS_SRCC_report_only"] = rho
            row["pooled_EA_MOS_p"] = p
            row["pooled_EA_MOS_N"] = n
            rows.append(row)
    return pd.DataFrame(rows)


def val_norm_stats(val_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ["free", "family"]:
        for variant in ["soft", "hard"]:
            col = f"E_A_{arm}_{variant}"
            for dataset, group in val_scores.groupby("dataset", sort=True):
                rows.append(
                    {
                        "arm": arm,
                        "variant": variant,
                        "dataset": dataset,
                        "N_val": int(len(group)),
                        "mean_E_A": float(group[col].mean()),
                        "std_E_A": float(group[col].std(ddof=0)),
                    }
                )
    return pd.DataFrame(rows)


def build_holdout_scores(holdout_scores: pd.DataFrame, norm_stats: pd.DataFrame) -> pd.DataFrame:
    taskd = pd.read_csv(TASKD_HOLDOUT)
    base = taskd[taskd["dataset"].eq(KADID)][["image_id", "E_H", "E_A", "Qz_AmH", "Qz_HmA", "z_H", "z_A"]].rename(
        columns={"E_A": "S0_E_A_ep05", "z_A": "S0_z_A_ep05", "Qz_AmH": "S0_Qz_AmH_ep05", "Qz_HmA": "S0_Qz_HmA_ep05"}
    )
    out = holdout_scores.merge(base, on="image_id", how="left", validate="one_to_one")
    stats = norm_stats.set_index(["arm", "variant", "dataset"])
    for arm in ["free", "family"]:
        for variant in ["soft", "hard"]:
            ea_col = f"E_A_{arm}_{variant}"
            st = stats.loc[(arm, variant, KADID)]
            z_col = f"z_A_{arm}_{variant}"
            out[z_col] = (out[ea_col] - float(st["mean_E_A"])) / float(st["std_E_A"])
            out[f"Qz_AmH_{arm}_{variant}"] = out[z_col] - out["z_H"]
            out[f"Qz_HmA_{arm}_{variant}"] = out["z_H"] - out[z_col]
    return out


def holdout_metrics(holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ["free", "family"]:
        for variant in ["soft", "hard"]:
            ea = f"E_A_{arm}_{variant}"
            q_amh = f"Qz_AmH_{arm}_{variant}"
            q_hma = f"Qz_HmA_{arm}_{variant}"
            for metric, col, target, dtype in [
                ("E_A vs MOS", ea, "mos_or_dmos", None),
                ("Qz_AmH vs MOS", q_amh, "mos_or_dmos", None),
                ("Qz_HmA vs MOS", q_hma, "mos_or_dmos", None),
                ("Blur E_A--sev", ea, "severity_or_level", "Gaussian blur"),
                ("Sharpen E_A--sev", ea, "severity_or_level", "High sharpen"),
                ("Pixelate E_A--sev", ea, "severity_or_level", "Pixelate"),
            ]:
                sub = holdout if dtype is None else holdout[holdout["distortion_type"].eq(dtype)]
                sr, srp, n = corr(sub[col], sub[target])
                pr, prp, _ = pcorr(sub[col], sub[target])
                rows.append(
                    {
                        "arm": arm,
                        "variant": variant,
                        "metric": metric,
                        "subset": "KADID holdout" if dtype is None else f"KADID holdout {dtype}",
                        "N": n,
                        "SRCC": sr,
                        "SRCC_p": srp,
                        "Pearson": pr,
                        "Pearson_p": prp,
                    }
                )
    return pd.DataFrame(rows)


def comparative_table(holdout_metrics_df: pd.DataFrame) -> pd.DataFrame:
    taskd = pd.read_csv(TASKD_HOLDOUT)
    kadid = taskd[taskd["dataset"].eq(KADID)]
    day9 = pd.read_csv(DAY9_HOLDOUT_METRICS)
    rows = []

    def s0_value(metric: str) -> float:
        if metric == "Holdout Q_z^(A-H)":
            return corr(kadid["Qz_AmH"], kadid["mos_or_dmos"])[0]
        if metric == "Blur E_A--sev":
            sub = kadid[kadid["distortion_type"].eq("Gaussian blur")]
            return corr(sub["E_A"], sub["severity_or_level"])[0]
        if metric == "Sharpen E_A--sev":
            sub = kadid[kadid["distortion_type"].eq("High sharpen")]
            return corr(sub["E_A"], sub["severity_or_level"])[0]
        raise KeyError(metric)

    def day9_value(metric: str) -> float:
        wanted = {
            "Holdout Q_z^(A-H)": "Qz_AmH_GMM vs MOS",
            "Blur E_A--sev": "Blur E_A_GMM vs severity",
            "Sharpen E_A--sev": "Sharpen E_A_GMM vs severity",
        }[metric]
        sub = day9[(day9["epoch_tag"].eq("ep05")) & (day9["energy_variant"].eq("soft")) & (day9["metric"].eq(wanted))]
        return float(sub.iloc[0]["SRCC"]) if not sub.empty else math.nan

    def day10_value(arm: str, metric: str) -> float:
        wanted = {
            "Holdout Q_z^(A-H)": "Qz_AmH vs MOS",
            "Blur E_A--sev": "Blur E_A--sev",
            "Sharpen E_A--sev": "Sharpen E_A--sev",
        }[metric]
        sub = holdout_metrics_df[(holdout_metrics_df["arm"].eq(arm)) & (holdout_metrics_df["variant"].eq("soft")) & (holdout_metrics_df["metric"].eq(wanted))]
        return float(sub.iloc[0]["SRCC"]) if not sub.empty else math.nan

    for arm_label, source in [
        ("S0 ep05 (Day-8 join)", "s0"),
        ("Day-9 GMM ep05 K=8", "day9"),
        ("Day-10 free K=5 no KonIQ", "free"),
        ("Day-10 family K<=5 no KonIQ", "family"),
    ]:
        row = {"Arm": arm_label}
        for metric in ["Holdout Q_z^(A-H)", "Blur E_A--sev", "Sharpen E_A--sev"]:
            if source == "s0":
                row[metric] = s0_value(metric)
            elif source == "day9":
                row[metric] = day9_value(metric)
            else:
                row[metric] = day10_value(source, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    require_inputs()

    train_ids_all = pd.read_csv(TRAIN_IDS)
    train_mu_all = np.load(TRAIN_MU)
    train_mask = train_ids_all["dataset"].isin(DATASETS).to_numpy()
    train_ids = train_ids_all[train_mask].copy().reset_index(drop=True)
    train_mu = train_mu_all[train_mask]
    if train_ids["dataset"].eq("KONIQ-10k").any():
        raise RuntimeError("KonIQ rows leaked into Day 10 train set")

    train_ids.to_csv(OUT_DIR / "mu_train_ids_kadid_tid_only.csv", index=False)
    np.save(OUT_DIR / "mu_train_kadid_tid_only.npy", train_mu.astype(np.float32))

    family_map = build_family_map(train_ids)
    family_map.to_csv(OUT_DIR / "family_map.csv", index=False)

    free_path, free_meta = fit_free(train_mu)
    family_path, family_meta = fit_family(train_mu, train_ids, family_map)
    models = {"free": free_path, "family": family_path}

    val_ids = pd.read_csv(VAL_IDS)
    val_mu = np.load(VAL_MU)
    holdout_ids = pd.read_csv(HOLDOUT_IDS)
    holdout_mu = np.load(HOLDOUT_MU)
    if val_ids["dataset"].eq("KONIQ-10k").any() or holdout_ids["dataset"].eq("KONIQ-10k").any():
        raise RuntimeError("KonIQ rows leaked into Day 10 val/holdout scoring")

    val_scores = add_arm_scores(val_ids, val_mu, models)
    val_scores.to_csv(OUT_DIR / "val_scores.csv", index=False)
    vm = val_metrics(val_scores)
    vm.to_csv(OUT_DIR / "val_metrics.csv", index=False)
    norm = val_norm_stats(val_scores)
    norm.to_csv(OUT_DIR / "val_normalization_stats.csv", index=False)

    holdout_raw = add_arm_scores(holdout_ids, holdout_mu, models)
    holdout_scores = build_holdout_scores(holdout_raw, norm)
    holdout_scores.to_csv(OUT_DIR / "holdout_scores.csv", index=False)
    hm = holdout_metrics(holdout_scores)
    hm.to_csv(OUT_DIR / "holdout_metrics.csv", index=False)
    comp = comparative_table(hm)
    comp.to_csv(OUT_DIR / "comparative_table.csv", index=False)

    fit_log = pd.DataFrame([free_meta, family_meta])
    fit_log.to_csv(OUT_DIR / "fit_log.csv", index=False)
    fit_lines = [
        "Day 10 fit log",
        "Train set: KADID+TID train only; all KonIQ rows removed before fitting.",
        f"Train N: {len(train_ids)}",
        f"Latent dim: {train_mu.shape[1]}",
        f"Seed: {SEED}",
        "",
        fit_log.to_string(index=False),
        "",
        "Family counts:",
        train_ids.assign(family_id=train_ids["distortion_type"].map(dict(zip(family_map["distortion_type"], family_map["family_id_final"])))).groupby("family_id").size().to_string(),
        "",
    ]
    (OUT_DIR / "fit_log.txt").write_text("\n".join(fit_lines))

    summary = {
        "task": "Day 10 - KADID+TID-only GMM, no KonIQ images",
        "output_dir": str(OUT_DIR),
        "seed": SEED,
        "train_N_kadid_tid_only": int(len(train_ids)),
        "train_dataset_counts": {str(k): int(v) for k, v in train_ids["dataset"].value_counts().to_dict().items()},
        "val_dataset_counts": {str(k): int(v) for k, v in val_ids["dataset"].value_counts().to_dict().items()},
        "holdout_dataset_counts": {str(k): int(v) for k, v in holdout_ids["dataset"].value_counts().to_dict().items()},
        "koniq_used": False,
        "models": {"free": free_meta, "family": family_meta},
        "outputs": {
            "free_model": str(free_path),
            "family_model": str(family_path),
            "family_map": str(OUT_DIR / "family_map.csv"),
            "fit_log": str(OUT_DIR / "fit_log.txt"),
            "val_scores": str(OUT_DIR / "val_scores.csv"),
            "val_metrics": str(OUT_DIR / "val_metrics.csv"),
            "holdout_scores": str(OUT_DIR / "holdout_scores.csv"),
            "holdout_metrics": str(OUT_DIR / "holdout_metrics.csv"),
            "comparative_table": str(OUT_DIR / "comparative_table.csv"),
            "report": str(OUT_DIR / "day10_report.txt"),
            "summary": str(OUT_DIR / "Day10_summary.json"),
        },
    }
    (OUT_DIR / "Day10_summary.json").write_text(json.dumps(summary, indent=2))

    report = [
        "Day 10 completed",
        "",
        "Important: KonIQ-10k images/rows were not used for Day 10 fit or scoring.",
        f"Train set used for fit: {len(train_ids)} rows = KADID+TID train only.",
        f"Val scoring set: {len(val_ids)} rows = KADID+TID val only.",
        f"Holdout scoring set: {len(holdout_ids)} rows = KADID holdout only.",
        "",
        "Fit summary:",
        fit_log.to_string(index=False),
        "",
        "Validation metrics:",
        vm.to_string(index=False),
        "",
        "KADID holdout metrics:",
        hm.to_string(index=False),
        "",
        "Comparative table:",
        comp.to_string(index=False),
        "",
    ]
    (OUT_DIR / "day10_report.txt").write_text("\n".join(report))
    (OUT_DIR / "DONE.txt").write_text(
        "Day 10 DONE\n"
        "No KonIQ images used.\n"
        f"Output dir: {OUT_DIR}\n"
        f"free model: {free_path}\n"
        f"family model: {family_path}\n"
    )

    print(f"Day 10 complete: {OUT_DIR}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
