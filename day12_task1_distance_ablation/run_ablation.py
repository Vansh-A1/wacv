#!/usr/bin/env python3
"""Run the frozen distance ablation in order, with an explicit holdout gate."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, rankdata, spearmanr
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DAY = ROOT / "day11_day12"
sys.path.insert(0, str(DAY))
sys.path.insert(0, str(HERE))
import day11_rankall_pipeline as base
from energies import (DISTANCES, ELIGIBLE, EPS, SIGMA_T_MAX, d1_validity_probe,
                      elbo_style, latent_energies, severity_selection)

S0 = HERE / "stage0_controls"
S1 = HERE / "stage1_distance_ablation"
S2 = HERE / "stage2_normalization"
S3 = HERE / "stage3_validation_selection"
S4 = HERE / "stage4_holdout_evaluation"
OLD = DAY / "task3_correction/stage0_corrected_baseline"
STAGES = (S0, S1, S2, S3, S4)
KEY = ["dataset", "image_id"]
POLES = ("EA", "EH")
LOCKED = {"blur": "Gaussian blur", "lens": "Lens blur",
          "sharpen": "High sharpen", "pixelate": "Pixelate"}
TID_README = Path("/data/projectwork/swati_mam/new model data/tid2013/tid2013/readme")
TID_NAMES = (
    "Additive Gaussian noise", "Noise in color components", "Spatially correlated noise",
    "Masked noise", "High frequency noise", "Impulse noise", "Quantization noise",
    "Gaussian blur", "Image denoising", "JPEG compression", "JPEG2000 compression",
    "JPEG transmission errors", "JPEG2000 transmission errors", "Non eccentricity pattern noise",
    "Local block-wise distortions", "Mean shift", "Contrast change", "Color saturation change",
    "Multiplicative Gaussian noise", "Comfort noise", "Lossy compression of noisy images",
    "Color quantization with dither", "Chromatic aberrations", "Sparse sampling and reconstruction",
)
BATCH = 64
SEED = 42
BOOTSTRAPS = 5000


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def log(message):
    print(f"[{now()}] {message}", flush=True)


def finite(x, label):
    if not np.isfinite(np.asarray(x)).all():
        raise ValueError(f"Non-finite values in {label}")


def corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite(x, "correlation x")
    finite(y, "correlation y")
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"N": len(x), "SRCC": None, "Pearson": None,
                "SRCC_p": None, "Pearson_p": None}
    sr, pr = spearmanr(x, y), pearsonr(x, y)
    return {"N": len(x), "SRCC": float(sr.statistic), "Pearson": float(pr.statistic),
            "SRCC_p": float(sr.pvalue), "Pearson_p": float(pr.pvalue)}


def read_scores(path):
    return pd.read_csv(path, keep_default_na=False)


def save_csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.17g")


def join_exact(left, right, on=KEY):
    result = left.merge(right, on=on, how="outer", validate="one_to_one", indicator=True,
                        sort=False)
    if not result["_merge"].eq("both").all():
        raise ValueError(f"Incomplete join: {result['_merge'].value_counts().to_dict()}")
    return result.drop(columns="_merge")


def mark(stage, files):
    write_json(stage / "DONE.json", {"completed_at": now(), "files": {
        str(Path(p).relative_to(HERE)): sha(p) for p in files}})


def check_done(stage):
    path = stage / "DONE.json"
    if not path.exists():
        return False
    for rel, expected in read_json(path)["files"].items():
        p = HERE / rel
        if not p.exists() or sha(p) != expected:
            raise RuntimeError(f"Completed stage output changed: {p}")
    return True


def require(stage):
    if not check_done(stage):
        raise RuntimeError(f"Run {stage.name} first")


def protocol():
    return read_json(S0 / "protocol.json")


def inputs():
    p = protocol()
    for path, digest in p["source_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"Input changed after protocol lock: {path}")
    return p


def stage0():
    if check_done(S0):
        inputs()
        log("Stage 0 controls verified and reused.")
        return
    log("Stage 0: check corrected controls, checkpoint identities, and split manifests.")
    chosen = read_json(OLD / "selected_checkpoints.json")
    m = base.load_manifest()
    m = m[m.dataset.isin(["KADID-10k", "TID2013"])].copy()
    m["distortion_type_original"] = m["distortion_type"]
    mapping = {f"type_{i:02d}": name for i, name in enumerate(TID_NAMES, 1)}
    is_tid = m.dataset.eq("TID2013")
    if not m.loc[is_tid, "distortion_type"].isin(mapping).all():
        raise ValueError("Unknown TID distortion code")
    m.loc[is_tid, "distortion_type"] = m.loc[is_tid, "distortion_type"].map(mapping)
    readme = TID_README.read_text(errors="replace")
    if "8       Gaussian blur" not in readme:
        raise ValueError("Cannot confirm the TID Gaussian blur mapping")
    assert not m.duplicated(KEY).any()
    assert m.severity_or_level.isin([1, 2, 3, 4, 5]).all()
    assert m.groupby(["dataset", "ref_id"]).split.nunique().max() == 1
    assert not m.distorted_path.duplicated().any()
    # Check filenames against reference and severity metadata, without opening images.
    for row in m.itertuples():
        parts = row.image_id.split("_")
        assert len(parts) == 3 and parts[0].upper() == row.ref_id.upper()
        assert int(parts[2]) == int(row.severity_or_level)
        assert Path(row.distorted_path).stem == row.image_id
        assert Path(row.distorted_path).is_file()
    use = {"train": m[m.split.eq("train")].copy(), "val": m[m.split.eq("val")].copy(),
           "holdout": m[m.split.eq("holdout") & m.dataset.eq("KADID-10k")].copy()}
    assert [len(use[k]) for k in use] == [9040, 1860, 1625]
    sources = [base.SPLIT_CSV, ROOT / "score.py", ROOT / "external/model.py",
               ROOT / "external/dataloader.py", DAY / "day11_rankall_pipeline.py",
               TID_README, HERE / "energies.py"]
    poles = {}
    for pole, label, expected_epoch in [("EA", "ranked_EA", 5), ("EH", "ranked_EH", 20)]:
        cp = Path(chosen[label]["ckpt_path"])
        c = torch.load(cp, map_location="cpu", weights_only=False)
        cfg = c["config"]
        assert int(c["epoch"]) == expected_epoch == int(chosen[label]["stored_epoch"])
        assert float(cfg["lambda_rank"]) == float(chosen[label]["lambda"]) == 0.1
        assert tuple(c["mu_ref"].shape) == tuple(c["Sigma_ref"].shape) == (100,)
        finite(c["mu_ref"], "mu_ref")
        finite(c["Sigma_ref"], "Sigma_ref")
        assert (c["Sigma_ref"] > 0).all()
        init = base.INIT_CKPT if pole == "EA" else base.EH_CKPT
        parent = torch.load(init, map_location="cpu", weights_only=False)
        ref_matches = all(torch.equal(c[k], parent[k]) for k in ("mu_ref", "Sigma_ref"))
        if not ref_matches:
            raise ValueError(f"{pole} stored reference differs from its initialization")
        ref_file = S0 / f"{pole}_stored_reference.pt"
        torch.save({k: c[k] for k in ("mu_ref", "Sigma_ref")}, ref_file)
        poles[pole] = {"checkpoint": str(cp), "checkpoint_sha256": sha(cp),
                       "epoch": expected_epoch, "lambda": 0.1,
                       "checkpoint_kind": c["checkpoint_kind"], "reference_file": str(ref_file),
                       "reference_parent": str(init), "reference_matches_parent": ref_matches,
                       "reference_semantics": "Diagonal variance across reference encoder means; "
                            "not mean posterior variance and not total mixture covariance",
                       "reference_summary": {k: {"shape": list(c[k].shape),
                            "min": c[k].min().item(), "max": c[k].max().item(),
                            "mean": c[k].mean().item()} for k in ("mu_ref", "Sigma_ref")}}
        sources.extend([cp, init])
    copied = []
    for file in sorted(OLD.iterdir()):
        if file.is_file():
            shutil.copy2(file, S0 / file.name)
            assert sha(file) == sha(S0 / file.name)
            sources.append(file)
            copied.append(S0 / file.name)
    # The source checkpoint JSON is kept verbatim; the protocol is the new identity record.
    manifests = []
    for split, frame in use.items():
        frame = frame.reset_index(drop=True)
        file = S0 / f"{split}_manifest.csv"
        save_csv(frame, file)
        manifests.append(file)
    save_csv(pd.DataFrame([{"dataset": "TID2013", "source_type": k, "canonical_type": v,
                           "source": str(TID_README)} for k, v in mapping.items()]),
             S0 / "distortion_mapping.csv")
    p = {"locked_at": now(), "poles": poles, "training": False, "seed": SEED,
         "datasets": ["KADID-10k", "TID2013"], "holdout_dataset": "KADID-10k",
         "counts": {k: len(v) for k, v in use.items()},
         "source_hashes": {str(f): sha(f) for f in sources},
         "manifest_hashes": {str(f): sha(f) for f in manifests},
         "preprocessing": "Inherited PathDataset: RGB, upscale short side only if <256, "
                          "center_crop 256, float32 [0,1]; batch size 64; original manifest order",
         "reconstruction": "Posterior mean with the existing skip-connected decoder; pixel L1",
         "reference_policy": "Use each exact checkpoint's embedded reference without refit",
         "train_z": "Pooled KADID+TID train, per pole and D4 component, ddof=0, denominator std+1e-8",
         "validation_z": "Per dataset, pole and candidate, val only, ddof=0; KADID stats for holdout",
         "selection": {"EH": "rho_blur+rho_lens+rho_sharpen+rho_pixelate",
                       "EA": "-rho_blur-rho_lens-max(0,rho_sharpen)-max(0,rho_pixelate)",
                       "scope": "KADID+TID val. Gaussian blur includes TID type_08; other "
                                "locked types available only in KADID.",
                       "tie_break": "smallest D id", "MOS_used": False,
                       "eligible": list(ELIGIBLE), "D1": d1_validity_probe()},
         "relative_score": "z(EA selected independently)-z(EH selected independently)",
         "bootstrap": {"n": BOOTSTRAPS, "seed": SEED, "cluster": "ref_id",
                       "paired": True, "interval": "percentile 2.5 and 97.5", "unit": "whole reference"},
         "pair_rule": "All ordered mild<severe pairs within dataset,ref_id,type; strict accuracy "
                      "plus tie count and half-credit accuracy",
         "environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    write_json(S0 / "protocol.json", p)
    (S0 / "stage0_checks.txt").write_text(
        "Corrected D0 controls copied and hashed.\nEA lambda 0.1 epoch 5; EH lambda 0.1 epoch 20.\n"
        "Train 9040; validation 1860; KADID holdout 1625. Dataset/ref split overlap: zero.\n"
        "Full candidate selection and uncertainty protocol was locked before new inference.\n")
    mark(S0, manifests + copied + [S0 / "protocol.json", S0 / "distortion_mapping.csv",
                                 S0 / "EA_stored_reference.pt", S0 / "EH_stored_reference.pt",
                                 S0 / "stage0_checks.txt"])


def posterior_mean(self, mu, logvar):
    return mu


def extract(pole, split, folder, device):
    """One encoder/decoder pass per pole and split; cache scores and full reconstructions."""
    spec = protocol()["poles"][pole]
    folder.mkdir(parents=True, exist_ok=True)
    frame = read_scores(S0 / f"{split}_manifest.csv")
    key = {"checkpoint_sha256": spec["checkpoint_sha256"], "epoch": spec["epoch"],
           "lambda": spec["lambda"], "pole": pole, "split": split,
           "manifest_sha256": sha(S0 / f"{split}_manifest.csv"),
           "energies_sha256": sha(HERE / "energies.py"), "batch_size": BATCH,
           "reconstruction": "posterior_mean_float32"}
    signature = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    done_path = folder / "cache.json"
    if done_path.exists():
        saved = read_json(done_path)
        if saved["signature"] != signature:
            raise RuntimeError(f"Cache identity mismatch: {folder}")
        if sha(folder / "posterior.npz") != saved["posterior_sha256"]:
            raise RuntimeError(f"Posterior cache changed: {folder}")
        arr = np.load(folder / "reconstruction.npy", mmap_mode="r")
        assert arr.shape == (len(frame), 3, 256, 256) and arr.dtype == np.float32
        with np.load(folder / "posterior.npz", allow_pickle=False) as data:
            values = {k: data[k] for k in data.files}
        log(f"Reuse {pole} {split} cache ({len(frame)} images).")
        return frame, values, saved
    if split == "holdout":
        require(S3)
    log(f"Extract {pole} {split}: {len(frame)} images, checkpoint epoch {spec['epoch']}.")
    model, _, ref, sigma, size, dim = base.load_model(Path(spec["checkpoint"]), device)
    assert size == 256 and dim == 100
    model.requires_grad_(False)
    model.reparameterize = types.MethodType(posterior_mean, model)
    loader = DataLoader(base.PathDataset(frame, size), batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True)
    recons = np.lib.format.open_memmap(folder / "reconstruction.npy", mode="w+", dtype="float32",
                                      shape=(len(frame), 3, size, size))
    buffers = {k: [] for k in ["mu", "logvar", "reconstruction_l1", "D0", "D1", "D2", "D3"]}
    started = time.monotonic()
    with torch.inference_mode():
        for i, (images, indices) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            recon, mu, lv = model(images)
            vals = latent_energies(mu, lv, ref, sigma)
            vals.update(mu=mu, logvar=lv, reconstruction_l1=(recon - images).abs().mean((1, 2, 3)))
            for k, value in vals.items():
                arr = value.cpu().numpy()
                finite(arr, f"{pole}/{split}/{k}")
                buffers[k].append(arr)
            arr = recon.cpu().numpy()
            finite(arr, "reconstruction")
            recons[indices.numpy()] = arr
            if (i + 1) % 40 == 0 or (i + 1) == len(loader):
                log(f"  {pole} {split}: {min((i+1)*BATCH,len(frame))}/{len(frame)}, "
                    f"{time.monotonic()-started:.1f}s")
    recons.flush()
    del recons
    values = {k: np.concatenate(v) for k, v in buffers.items()}
    values.update(mu_ref=ref.cpu().numpy(), Sigma_ref=sigma.cpu().numpy())
    np.savez_compressed(folder / "posterior.npz", **values)
    info = {**key, "signature": signature, "N": len(frame), "latent_dim": dim,
            "device": str(device), "completed_at": now(),
            "posterior_sha256": sha(folder / "posterior.npz"),
            "reconstruction_bytes": (folder / "reconstruction.npy").stat().st_size,
            "logvar_min": values["logvar"].min(), "logvar_max": values["logvar"].max(),
            "posterior_variance_min": np.exp(values["logvar"].astype(float)).min(),
            "posterior_variance_max": np.exp(values["logvar"].astype(float)).max(),
            "D1_fraction_variances_capped_at_one": (values["logvar"] > 0).mean(),
            "all_scores_finite": True}
    write_json(done_path, info)
    del model
    torch.cuda.empty_cache()
    return frame, values, info


def cache_manifest(pole, split, frame, folder, info):
    out = frame.copy()
    out["pole"] = pole
    out["checkpoint_path"] = protocol()["poles"][pole]["checkpoint"]
    out["checkpoint_sha256"] = info["checkpoint_sha256"]
    out["epoch"] = info["epoch"]
    out["lambda"] = info["lambda"]
    out["cache_row"] = np.arange(len(frame))
    out["posterior_file"] = str(folder / "posterior.npz")
    out["reconstruction_file"] = str(folder / "reconstruction.npy")
    return out


def raw_frame(frame, pole, values, stats):
    out = frame.copy()
    for k in DISTANCES:
        out[f"{pole}_{k}"] = (elbo_style(values["D0"].astype(float),
                                      values["reconstruction_l1"].astype(float), stats)
                                  if k == "D4" else values[k])
    out[f"{pole}_reconstruction_l1"] = values["reconstruction_l1"]
    return out


def combine_poles(frames):
    return join_exact(frames["EA"], frames["EH"][KEY + [c for c in frames["EH"] if c.startswith("EH_")]])


def stage1(device):
    require(S0)
    if check_done(S1):
        log("Stage 1 train/validation candidates verified and reused.")
        return
    log("Stage 1: cache posteriors and deterministic reconstructions; compute D0-D4.")
    definitions = {
        "source_score_py": str(ROOT / "score.py"), "source_sha256": sha(ROOT / "score.py"),
        "exact_existing_function": inspect.getsource(base.sz_from_stats),
        "eps": EPS, "D1_sigma_t_max": SIGMA_T_MAX, "covariance": protocol()["poles"],
        "D0": {"formula": "sqrt(sum_j((mu_ref_j-mu_x_j)^2/(Sigma_ref_j+1e-8)))",
               "implementation": "Exact score.sz_from_stats, mu_only=True", "eligible": True},
        "D1": {"formula": "sqrt(sum_j((mu_ref_j-mu_x_j)^2/(Sigma_ref_j+min(exp(logvar_x_j),1)+1e-8)))",
               "implementation": "Exact score.sz_from_stats, mu_only=False",
               "eligible": False, "validity": d1_validity_probe()},
        "D2": {"formula": "sum_j((mu_x_j-mu_ref_j)^2+(sqrt(vx_j)-sqrt(vr_j))^2)",
               "squared": True, "eligible": True},
        "D3": {"formula": "sum_j(delta_mu_j^2/bar_v_j)/8 + "
                           "sum_j(log(bar_v_j)-(log(vx_j)+log(vr_j))/2)/2",
               "bar_v": "(vx+vr)/2", "eligible": True},
        "D4": {"formula": "(D0-train_mean_D0)/(train_std_D0+eps) + "
                           "(L1_recon-train_mean_L1)/(train_std_L1+eps)",
               "reconstruction_loss": "mean absolute error across RGB pixels, recon at posterior mean",
               "fit_scope": "9040 KADID+TID train images, per pole, ddof=0", "eligible": True,
               "interpretation": "ELBO-style diagnostic, not a calibrated likelihood bound"},
        "D2_D3_variances": "vx=max(exp(logvar),1e-8), vr=max(diag(Sigma_ref),1e-8); "
                           "no upper cap; actual stored Sigma_ref is a 100-element variance vector",
        "reference_caveat": "Variance across encoder means is not posterior uncertainty. "
                            "D2/D3 compare an image posterior to the specified stored Gaussian proxy.",
        "reconstruction_model": "Existing skip-connected decoder with reparameterize(mu,lv)=mu",
    }
    write_json(S1 / "distance_definitions.json", definitions)
    caches, score_frames, stats_rows, train_component_stats = [], {"train": {}, "val": {}}, [], {}
    for pole in POLES:
        train_folder = S1 / "latent_cache" / pole / "train"
        train, values, info = extract(pole, "train", train_folder, device)
        stats = {k: {"mean": float(values[k].astype(float).mean()),
                      "std": float(values[k].astype(float).std(ddof=0)), "N": len(train)}
                 for k in ("D0", "reconstruction_l1")}
        assert all(v["std"] > 0 for v in stats.values())
        train_component_stats[pole] = stats
        caches.append(cache_manifest(pole, "train", train, train_folder, info))
        score_frames["train"][pole] = raw_frame(train, pole, values, stats)
        for k in list(DISTANCES) + ["reconstruction_l1"]:
            x = score_frames["train"][pole][f"{pole}_{k}"].to_numpy(float)
            stats_rows.append({"pole": pole, "distance": k, "split": "train",
                               "datasets": "KADID-10k+TID2013", "N": len(x), "mean": x.mean(),
                               "std_population": x.std(ddof=0), "min": x.min(), "max": x.max(),
                               "used_in_D4": k in ("D0", "reconstruction_l1"), "epsilon": EPS})
        val_folder = S1 / "latent_cache" / pole / "val"
        val, val_values, val_info = extract(pole, "val", val_folder, device)
        score_frames["val"][pole] = raw_frame(val, pole, val_values, stats)
        caches.append(cache_manifest(pole, "val", val, val_folder, val_info))
    train = combine_poles(score_frames["train"])
    val = combine_poles(score_frames["val"])
    save_csv(train, S1 / "train_distance_scores.csv")
    save_csv(val, S1 / "val_distance_scores.csv")
    save_csv(pd.concat(caches, ignore_index=True), S1 / "latent_cache_manifest.csv")
    save_csv(pd.DataFrame(stats_rows), S1 / "train_distance_stats.csv")
    write_json(S1 / "D4_train_component_stats.json", train_component_stats)
    # Anchor the new D0 to each checkpoint's original validation cache.
    checks = []
    for pole, label, ecol in [("EA", "EA_l0.1_ep0005", "E_A"), ("EH", "EH_l0.1_ep0020", "E_H_ranked")]:
        old = read_scores(OLD / f"val_ranked_{label}.csv")
        compare = join_exact(val[val.dataset.eq("KADID-10k")][KEY + [f"{pole}_D0"]], old[KEY + [ecol]])
        diff = np.abs(compare[f"{pole}_D0"] - compare[ecol])
        tolerance = 0.002 if pole == "EA" else 0.01
        if diff.max() > tolerance:
            raise RuntimeError(f"D0 validation control mismatch {pole}: {diff.max()}")
        checks.append({"pole": pole, "split": "val", "N": len(compare),
                       "max_abs_error": diff.max(), "tolerance": tolerance})
    save_csv(pd.DataFrame(checks), S1 / "D0_val_control_checks.csv")
    mark(S1, [S1 / n for n in ["distance_definitions.json", "train_distance_scores.csv",
         "val_distance_scores.csv", "latent_cache_manifest.csv", "train_distance_stats.csv",
         "D4_train_component_stats.json", "D0_val_control_checks.csv"]])


def normalize(frame, stats):
    result = frame.copy()
    for row in stats.itertuples():
        mask = result.dataset.eq(row.dataset)
        if row.std_population <= 0:
            raise ValueError(f"Nonpositive validation std: {row}")
        # NumPy/Pandas scalar promotion can otherwise retain the GPU array's float32 dtype.
        raw = result.loc[mask, f"{row.pole}_{row.distance}"].to_numpy(dtype=np.float64)
        result.loc[mask, f"z_{row.pole}_{row.distance}"] = (raw - row.mean) / row.std_population
    for distance in DISTANCES:
        result[f"Qz_{distance}"] = result[f"z_EA_{distance}"] - result[f"z_EH_{distance}"]
    finite(result.filter(regex=r"^(z_|Qz_)").to_numpy(float), "normalized energies")
    return result


def stage2():
    require(S1)
    if check_done(S2):
        log("Stage 2 normalization verified and reused.")
        return
    log("Stage 2: fit and freeze validation-only normalizers.")
    val = read_scores(S1 / "val_distance_scores.csv")
    assert val.split.eq("val").all()
    rows = []
    for dataset, frame in val.groupby("dataset", sort=False):
        for pole in POLES:
            for distance in DISTANCES:
                x = frame[f"{pole}_{distance}"].to_numpy(float)
                rows.append({"dataset": dataset, "pole": pole, "distance": distance,
                             "mean": x.mean(), "std_population": x.std(ddof=0),
                             "N": len(x), "fit_split": "val", "ddof": 0})
    stats = pd.DataFrame(rows)
    normed = normalize(val, stats)
    save_csv(stats, S2 / "validation_normalization_stats.csv")
    save_csv(normed, S2 / "val_normalized_scores.csv")
    write_json(S2 / "normalization_provenance.json", {
        "frozen_at": now(), "raw_validation_sha256": sha(S1 / "val_distance_scores.csv"),
        "formula": "z=(energy-dataset_validation_mean)/dataset_validation_population_std",
        "holdout_statistics_used": False, "stats": stats.to_dict("records")})
    mark(S2, [S2 / "validation_normalization_stats.csv", S2 / "val_normalized_scores.csv",
              S2 / "normalization_provenance.json"])


def stage3():
    require(S2)
    if check_done(S3):
        log("Stage 3 frozen choices verified and reused.")
        return
    log("Stage 3: select each pole using KADID+TID validation severity only.")
    check_model_gradients()
    val = read_scores(S2 / "val_normalized_scores.csv")
    rows, per_dataset, winners = [], [], {}
    for pole in POLES:
        for distance in DISTANCES:
            correlations = {}
            row = {"pole": pole, "distance": distance, "eligible": distance in ELIGIBLE,
                   "checkpoint_path": protocol()["poles"][pole]["checkpoint"],
                   "epoch": protocol()["poles"][pole]["epoch"], "lambda": 0.1}
            for key, kind in LOCKED.items():
                sub = val[val.distortion_type.eq(kind)]
                result = corr(sub[f"{pole}_{distance}"], sub.severity_or_level)
                if result["SRCC"] is None:
                    raise RuntimeError(f"Missing selection correlation {pole} {distance} {kind}")
                correlations[key] = result["SRCC"]
                row[f"{key}_SRCC"], row[f"{key}_N"] = result["SRCC"], result["N"]
                for ds, group in sub.groupby("dataset"):
                    per_dataset.append({"pole": pole, "distance": distance, "dataset": ds,
                                        "distortion_type": kind,
                                        **corr(group[f"{pole}_{distance}"], group.severity_or_level)})
            row["selection_score"] = severity_selection(pole, correlations)
            report = corr(val[f"{pole}_{distance}"], val.mos_or_dmos)
            row["pooled_MOS_SRCC_report_only"] = report["SRCC"]
            row["pooled_MOS_Pearson_report_only"] = report["Pearson"]
            row["pooled_MOS_N_report_only"] = report["N"]
            row["selected"] = False
            rows.append(row)
        candidates = [r for r in rows if r["pole"] == pole and r["eligible"]]
        best = sorted(candidates, key=lambda r: (-r["selection_score"], r["distance"]))[0]
        best["selected"] = True
        winners[pole] = best.copy()
    table = pd.DataFrame(rows)
    save_csv(table, S3 / "val_distance_selection.csv")
    save_csv(pd.DataFrame(per_dataset), S3 / "val_selection_by_dataset.csv")
    q_rows = []
    for dataset, frame in [("pooled_report_only", val)] + list(val.groupby("dataset")):
        for distance in DISTANCES:
            q_rows.append({"dataset": dataset, "distance": distance, "use": "report_only",
                           **corr(frame[f"Qz_{distance}"], frame.mos_or_dmos)})
    save_csv(pd.DataFrame(q_rows), S3 / "val_relative_correlations_report_only.csv")
    selected = {"frozen_at": now(), "winners": winners,
                "EA_distance": winners["EA"]["distance"], "EH_distance": winners["EH"]["distance"],
                "relative_formula": f"z_EA_{winners['EA']['distance']}-z_EH_{winners['EH']['distance']}",
                "relative_selection": "Combine independently selected poles, no relative-score search",
                "MOS_used_for_selection": False, "holdout_opened": False,
                "validation_scores_sha256": sha(S2 / "val_normalized_scores.csv"),
                "normalization_stats_sha256": sha(S2 / "validation_normalization_stats.csv"),
                "checkpoint_hashes": {k: v["checkpoint_sha256"] for k, v in protocol()["poles"].items()}}
    write_json(S3 / "selected_distances.json", selected)
    for filename in ("val_distance_selection.csv", "selected_distances.json"):
        link = S1 / filename
        if not link.exists():
            link.symlink_to(Path("..") / S3.name / filename)
    selection_plot(table)
    mark(S3, [S3 / n for n in ["val_distance_selection.csv", "val_selection_by_dataset.csv",
         "val_relative_correlations_report_only.csv", "selected_distances.json", "selection_scores.png",
         "differentiability_checks.csv"]])
    log(f"Frozen: EA {selected['EA_distance']}; EH {selected['EH_distance']}; {selected['relative_formula']}.")


def check_model_gradients():
    frame = read_scores(S0 / "val_manifest.csv").iloc[:1].copy()
    train_stats = read_json(S1 / "D4_train_component_stats.json")
    rows = []
    for pole in POLES:
        spec = protocol()["poles"][pole]
        model, _, ref, sigma, size, _ = base.load_model(Path(spec["checkpoint"]), torch.device("cuda"))
        model.requires_grad_(False)
        targets = [model.enc1.weight, model.fc_mu.weight, model.fc_log_var.weight]
        for weight in targets:
            weight.requires_grad_(True)
        model.reparameterize = types.MethodType(posterior_mean, model)
        image, _ = base.PathDataset(frame, size)[0]
        image = image.unsqueeze(0).cuda()
        recon, mu, lv = model(image)
        scores = latent_energies(mu, lv, ref, sigma)
        scores["D4"] = elbo_style(scores["D0"], (recon-image).abs().mean((1,2,3)), train_stats[pole])
        for distance in DISTANCES:
            gradients = torch.autograd.grad(scores[distance].sum(), targets, retain_graph=True,
                                            allow_unused=True)
            row = {"pole": pole, "distance": distance, "split": "val",
                   "image_id": frame.iloc[0].image_id, "score": scores[distance].item()}
            for name, gradient in zip(("encoder_first_layer", "mean_head", "logvar_head"), gradients):
                row[f"{name}_gradient_norm"] = 0.0 if gradient is None else gradient.norm().item()
                if gradient is not None and not torch.isfinite(gradient).all():
                    raise RuntimeError(f"Nonfinite model gradient: {pole} {distance} {name}")
            assert row["encoder_first_layer_gradient_norm"] > 0 and row["mean_head_gradient_norm"] > 0
            rows.append(row)
        del model
    save_csv(pd.DataFrame(rows), S3 / "differentiability_checks.csv")


def selection_plot(table):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, pole in zip(axes, POLES):
        sub = table[table.pole.eq(pole)]
        colors = ["#087f8c" if r.selected else "#a6adb4" if r.eligible else "#d9d9d9"
                  for r in sub.itertuples()]
        bars = ax.bar(sub.distance, sub.selection_score, color=colors)
        for bar, r in zip(bars, sub.itertuples()):
            if not r.eligible:
                bar.set_hatch("///")
            ax.annotate(f"{r.selection_score:.3f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4 if r.selection_score >= 0 else -12), textcoords="offset points",
                        ha="center", fontsize=9)
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set_title(f"{pole}: validation severity selection")
        ax.set_ylabel("Selection score (higher is preferred)")
        ax.margins(y=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.02, "Teal: frozen winner. Hatched D1: diagnostic only, excluded by validity test.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(S3 / "selection_scores.png", dpi=180)
    plt.close(fig)


def arms(selected):
    rows = []
    for pole in POLES:
        for distance in DISTANCES:
            rows.append({"arm": f"{pole}_{distance}", "column": f"{pole}_{distance}", "pole": pole,
                         "distance": distance, "orientation_to_quality": 1 if pole == "EA" else -1,
                         "baseline": f"{pole}_D0", "selected": selected[f"{pole}_distance"] == distance,
                         "eligible": distance in ELIGIBLE})
    for distance in DISTANCES:
        rows.append({"arm": f"Qz_{distance}{distance}", "column": f"Qz_{distance}", "pole": "Q",
                     "distance": distance, "orientation_to_quality": 1,
                     "baseline": "Qz_D0D0", "selected": False, "eligible": distance in ELIGIBLE})
    rows.append({"arm": "Qz_selected", "column": "Qz_selected", "pole": "Q", "distance": "selected",
                 "orientation_to_quality": 1, "baseline": "Qz_D0D0", "selected": True, "eligible": True})
    return rows


def pair_indices(frame):
    mild, severe, kinds, refs = [], [], [], []
    for (_, ref, kind), group in frame.groupby(["dataset", "ref_id", "distortion_type"]):
        positions = group.index.to_numpy()
        levels = group.severity_or_level.to_numpy(float)
        a, b = np.where(levels[:, None] < levels[None, :])
        mild.extend(positions[a]); severe.extend(positions[b])
        kinds.extend([kind] * len(a)); refs.extend([ref] * len(a))
    return np.asarray(mild), np.asarray(severe), np.asarray(kinds), np.asarray(refs)


def holdout_metrics(frame, specifications):
    mos_rows, sev_rows, pair_rows = [], [], []
    a, b, kinds, refs = pair_indices(frame)
    assert len(a) == 3250
    for arm in specifications:
        values = frame[arm["column"]].to_numpy(float)
        mos_rows.append({**arm, "dataset": "KADID-10k", "split": "holdout",
                         **corr(values, frame.mos_or_dmos)})
        for kind, group in [("all_types_pooled", frame)] + list(frame.groupby("distortion_type")):
            sev_rows.append({**arm, "dataset": "KADID-10k", "distortion_type": kind,
                             **corr(group[arm["column"]], group.severity_or_level)})
        difference = (values[a] - values[b]) * arm["orientation_to_quality"]
        for kind in ["all_types_pooled"] + sorted(set(kinds)):
            mask = np.ones(len(a), dtype=bool) if kind == "all_types_pooled" else kinds == kind
            diff = difference[mask]
            ties = int((diff == 0).sum())
            correct = int((diff > 0).sum())
            pair_rows.append({**arm, "distortion_type": kind, "N_pairs": len(diff),
                              "correct": correct, "ties": ties, "wrong": int((diff < 0).sum()),
                              "strict_accuracy": correct / len(diff),
                              "accuracy_half_credit_ties": (correct + ties / 2) / len(diff),
                              "N_refs": len(set(refs[mask]))})
    return pd.DataFrame(mos_rows), pd.DataFrame(sev_rows), pd.DataFrame(pair_rows)


def row_correlation(x, y):
    """Pearson along images for arrays [replicate, image, arm] and [replicate, image]."""
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    numerator = (xc * yc[:, :, None]).sum(axis=1)
    denominator = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1)[:, None])
    return numerator / denominator


def bootstrap(frame, specifications):
    log(f"Stage 4: {BOOTSTRAPS} paired bootstrap resamples of whole reference clusters.")
    refs = sorted(frame.ref_id.unique())
    assert len(refs) == 13
    groups = np.asarray([np.flatnonzero(frame.ref_id.eq(ref).to_numpy()) for ref in refs])
    assert groups.shape == (13, 125)
    generator = np.random.default_rng(SEED)
    draws = generator.integers(0, len(refs), size=(BOOTSTRAPS, len(refs)))
    x = frame[[a["column"] for a in specifications]].to_numpy(float)
    y = frame.mos_or_dmos.to_numpy(float)
    pearson = np.empty((BOOTSTRAPS, len(specifications)))
    srcc = np.empty_like(pearson)
    for start in range(0, BOOTSTRAPS, 25):
        ids = groups[draws[start:start+25]].reshape(-1, len(frame))
        sample_x, sample_y = x[ids], y[ids]
        pearson[start:start+len(ids)] = row_correlation(sample_x, sample_y)
        srcc[start:start+len(ids)] = row_correlation(rankdata(sample_x, axis=1), rankdata(sample_y, axis=1))
        if (start+25) % 1000 == 0:
            log(f"  Bootstrap {start+25}/{BOOTSTRAPS}")
    finite(srcc, "bootstrap SRCC")
    finite(pearson, "bootstrap Pearson")
    np.savez_compressed(S4 / "bootstrap_replicates.npz", srcc=srcc, pearson=pearson,
                        reference_draws=draws, reference_ids=np.asarray(refs),
                        arm_names=np.asarray([a["arm"] for a in specifications]))
    lookup = {a["arm"]: i for i, a in enumerate(specifications)}
    rows = []
    for i, arm in enumerate(specifications):
        j = lookup[arm["baseline"]]
        actual = corr(x[:, i], y)
        baseline_actual = corr(x[:, j], y)
        for metric, reps in [("SRCC", srcc), ("Pearson", pearson)]:
            raw_diff = reps[:, i] - reps[:, j]
            oriented_diff = raw_diff * arm["orientation_to_quality"]
            ci = np.percentile(reps[:, i], [2.5, 97.5])
            diff_ci = np.percentile(raw_diff, [2.5, 97.5])
            oriented_ci = np.percentile(oriented_diff, [2.5, 97.5])
            point_diff = actual[metric] - baseline_actual[metric]
            rows.append({**arm, "metric": metric, "N_images": len(frame), "N_ref_clusters": len(refs),
                         "bootstrap_n": BOOTSTRAPS, "seed": SEED, "estimate_raw": actual[metric],
                         "ci95_low_raw": ci[0], "ci95_high_raw": ci[1],
                         "baseline_estimate_raw": baseline_actual[metric], "delta_raw": point_diff,
                         "delta_ci95_low_raw": diff_ci[0], "delta_ci95_high_raw": diff_ci[1],
                         "delta_quality_oriented": point_diff * arm["orientation_to_quality"],
                         "delta_ci95_low_quality_oriented": oriented_ci[0],
                         "delta_ci95_high_quality_oriented": oriented_ci[1],
                         "ci_excludes_zero_in_favorable_direction": bool(oriented_ci[0] > 0)})
    return pd.DataFrame(rows)


def stage4(device, rebuild_tables=False):
    require(S3)
    if check_done(S4) and not rebuild_tables:
        log("Stage 4 recorded holdout results verified; no repeat scoring.")
        return
    if rebuild_tables:
        for pole in POLES:
            if not (S4 / "latent_cache" / pole / "holdout" / "cache.json").exists():
                raise RuntimeError("Rebuilding tables requires both completed holdout caches")
        archive = S4 / "previous_tables" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        archive.mkdir(parents=True, exist_ok=False)
        for file in S4.iterdir():
            if file.is_file():
                shutil.copy2(file, archive / file.name)
        log(f"Rebuild derived tables from completed holdout caches; previous tables copied to {archive}.")
    selected_path = S3 / "selected_distances.json"
    selected_hash = sha(selected_path)
    selected = read_json(selected_path)
    record = S4 / "holdout_started.json"
    if record.exists():
        assert read_json(record)["selection_sha256"] == selected_hash
    else:
        write_json(record, {"started_at": now(), "selection_sha256": selected_hash,
                            "frozen_before_holdout": True, "N": 1625})
    log("Stage 4: open KADID holdout with the frozen distance choices.")
    components = read_json(S1 / "D4_train_component_stats.json")
    frames, caches = {}, []
    for pole in POLES:
        folder = S4 / "latent_cache" / pole / "holdout"
        frame, values, info = extract(pole, "holdout", folder, device)
        frames[pole] = raw_frame(frame, pole, values, components[pole])
        caches.append(cache_manifest(pole, "holdout", frame, folder, info))
    raw = combine_poles(frames)
    stats = read_scores(S2 / "validation_normalization_stats.csv")
    frame = normalize(raw, stats)
    frame["EA_selected"] = frame[f"EA_{selected['EA_distance']}"]
    frame["EH_selected"] = frame[f"EH_{selected['EH_distance']}"]
    frame["Qz_selected"] = frame[f"z_EA_{selected['EA_distance']}"] - frame[f"z_EH_{selected['EH_distance']}"]
    old = read_scores(OLD / "corrected_ranked_dual_holdout.csv")
    old_cols = ["ranked_EA_raw", "ranked_EH_raw", "z_ranked_EA", "z_ranked_EH",
                "Qz_uniform_rankedEA_minus_rankedEH"]
    old = old[KEY + old_cols].rename(columns={k: f"stage0_{k}" for k in old_cols})
    frame = join_exact(frame, old)
    checks = []
    for col, control, tolerance in [
        ("EA_D0", "ranked_EA_raw", 1e-9), ("EH_D0", "ranked_EH_raw", 1e-9),
        ("z_EA_D0", "z_ranked_EA", 1e-9), ("z_EH_D0", "z_ranked_EH", 1e-9),
        ("Qz_D0", "Qz_uniform_rankedEA_minus_rankedEH", 1e-9)]:
        error = np.abs(frame[col] - frame[f"stage0_{control}"])
        if error.max() > tolerance:
            raise RuntimeError(f"Stage0 D0D0 mismatch: {col} {error.max()}")
        checks.append({"column": col, "control": control, "max_abs_error": error.max(),
                       "mean_abs_error": error.mean(), "tolerance": tolerance})
    for col in [c for c in frame if c.startswith(("EA_", "EH_", "z_", "Qz_"))]:
        finite(frame[col], col)
    save_csv(frame, S4 / "holdout_distance_scores.csv")
    save_csv(pd.concat(caches, ignore_index=True), S4 / "holdout_latent_cache_manifest.csv")
    save_csv(pd.DataFrame(checks), S4 / "D0_stage0_control_checks.csv")
    specifications = arms(selected)
    mos, severity, pairs = holdout_metrics(frame, specifications)
    save_csv(mos, S4 / "holdout_distance_correlations.csv")
    save_csv(severity, S4 / "holdout_severity_by_distortion.csv")
    save_csv(pairs, S4 / "holdout_pair_order_accuracy.csv")
    boot = bootstrap(frame, specifications)
    save_csv(boot, S4 / "holdout_bootstrap_confidence_intervals.csv")
    for filename in ("holdout_distance_scores.csv", "holdout_distance_correlations.csv"):
        link = S1 / filename
        if not link.exists():
            link.symlink_to(Path("..") / S4.name / filename)
    assert sha(selected_path) == selected_hash
    comparison_plot(mos, boot, selected)
    write_report(mos, severity, pairs, boot, selected)
    write_json(S4 / "holdout_completed.json", {"completed_at": now(),
               "selection_sha256": selected_hash, "scored_images_per_pole": 1625,
               "holdout_norm_fitted": False, "retraining": False})
    mark(S4, [S4 / n for n in ["holdout_distance_scores.csv", "holdout_distance_correlations.csv",
         "holdout_severity_by_distortion.csv", "holdout_pair_order_accuracy.csv",
         "holdout_bootstrap_confidence_intervals.csv", "bootstrap_replicates.npz",
         "D0_stage0_control_checks.csv", "holdout_latent_cache_manifest.csv", "holdout_started.json",
         "holdout_completed.json", "holdout_comparison.png", "stage4_report.txt"]])


def comparison_plot(mos, boot, selected):
    names = ["EA_D0", f"EA_{selected['EA_distance']}", "EH_D0", f"EH_{selected['EH_distance']}",
             "Qz_D0D0", "Qz_selected"]
    labels = ["EA D0", "EA selected", "-EH D0", "-EH selected", "Q D0D0", "Q selected"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    vals, low, high = [], [], []
    for arm in names:
        row = boot[boot.arm.eq(arm) & boot.metric.eq("SRCC")].iloc[0]
        sign = row.orientation_to_quality
        estimate = row.estimate_raw * sign
        ends = sorted([row.ci95_low_raw * sign, row.ci95_high_raw * sign])
        vals.append(estimate); low.append(max(0, estimate-ends[0])); high.append(max(0, ends[1]-estimate))
    bars = ax.bar(labels, vals, color=["#a6adb4", "#087f8c", "#a6adb4", "#c0547a", "#a6adb4", "#087f8c"],
                  yerr=[low, high], capsize=5, error_kw={"linewidth": 1})
    ax.set_ylabel("Quality-oriented SRCC vs MOS")
    ax.set_title("KADID holdout: frozen validation choices")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_ylim(min(0, min(np.asarray(vals)-low)-.07), max(np.asarray(vals)+high)+.09)
    for bar, value in zip(bars, vals):
        ax.annotate(f"{value:.3f}", (bar.get_x()+bar.get_width()/2, 0), ha="center",
                    va="bottom", xytext=(0, 5), textcoords="offset points", fontsize=9)
    fig.text(.5,.02,"95% intervals: 5,000 reference-clustered resamples. EH sign reversed for quality orientation.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    fig.savefig(S4 / "holdout_comparison.png", dpi=180)
    plt.close(fig)


def write_report(mos, severity, pairs, boot, selected):
    q = mos[mos.arm.eq("Qz_selected")].iloc[0]
    d0 = mos[mos.arm.eq("Qz_D0D0")].iloc[0]
    ci = boot[boot.arm.eq("Qz_selected") & boot.metric.eq("SRCC")].iloc[0]
    eh = mos[mos.arm.eq(f"EH_{selected['EH_distance']}")].iloc[0]
    improved = ci.delta_ci95_low_quality_oriented > 0
    lines = ["Day 12 Task 1: Frozen Distance Ablation", "", "Completed stages: 0, 1, 2, 3, 4.",
        "EA: ranked lambda=0.1 epoch=5; EH: ranked lambda=0.1 epoch=20.",
        "New training: none. No KonIQ image was loaded in this experiment.",
        "Read order: Photo 3 -> Photo 2 -> Photo 1 -> Photo 4 (overlapping pages).", "",
        "D1 validity: rejected for selection. Existing score.py responds to variance only by "
        "reweighting the mean difference. Equal means with unequal variances still score zero. "
        "All D1 values are retained and labeled diagnostic only.",
        "D2/D3: stored covariance is variance of reference encoder means. The image covariance "
        "is exp(logvar). This is the specified reference Gaussian proxy, not matched posterior "
        "uncertainty on both sides. Epsilon=1e-8; D2 is W2 squared.",
        "D4: fixed equal sum of train-standardized D0 and deterministic reconstruction L1. "
        "It is ELBO-style, not a true likelihood bound.", "",
        f"Frozen winners: EA={selected['EA_distance']}; EH={selected['EH_distance']}.",
        f"Q definition: {selected['relative_formula']}.",
        "Selection uses KADID+TID val severity only. Gaussian blur N=75 (KADID 60 + TID 15); "
        "lens, sharpen, pixelate N=60 each (KADID only; no exact TID counterpart).",
        "Pooled MOS correlations are report-only and mix dataset MOS scales.",
        "Outer normalization uses each dataset's validation; holdout uses KADID val N=1500, ddof=0.",
        "D4 component scaling uses pooled train N=9040. No holdout fitting.", "",
        "KADID holdout, N=1625:",
        f"  Corrected D0D0: SRCC={d0.SRCC:.9f}; Pearson={d0.Pearson:.9f}.",
        f"  Selected Q: SRCC={q.SRCC:.9f}; Pearson={q.Pearson:.9f}.",
        f"  Q SRCC difference: {ci.delta_raw:+.9f}; paired reference-clustered 95% CI "
        f"[{ci.delta_ci95_low_raw:+.9f}, {ci.delta_ci95_high_raw:+.9f}].",
        f"  Favorable difference with CI above zero: {improved}.",
        f"  Selected EH alone: raw SRCC={eh.SRCC:.9f}; quality-oriented SRCC={-eh.SRCC:.9f}.",
        "Confidence intervals use 5000 paired cluster draws, ref_id as cluster, seed=42, "
        "percentile 95% intervals and ranks recomputed after each draw.",
        "Intervals are descriptive: only 13 holdout reference clusters, and this holdout "
        "was used for earlier experiment reports. It is not a new external test set.",
        "Pair accuracy uses 3250 within-reference/type mild<severe pairs; ties are separate.", "",
        "All pole and relative MOS correlations:",
        mos[["arm", "eligible", "selected", "N", "SRCC", "Pearson"]].to_string(index=False), "",
        "Selected Q severity by distortion:",
        severity[severity.arm.eq("Qz_selected")][["distortion_type", "N", "SRCC", "Pearson"]].to_string(index=False),
        "", "Overall pair order accuracy:",
        pairs[pairs.distortion_type.eq("all_types_pooled")][["arm", "N_pairs", "strict_accuracy", "ties"]].to_string(index=False)]
    text = "\n".join(lines) + "\n"
    (S4 / "stage4_report.txt").write_text(text)
    (HERE / "FINAL_REPORT.md").write_text(text)


def verify():
    log("Verify all deliverables, identities, normalizers, formulas, metrics, and selection order.")
    p = inputs()
    for stage in STAGES:
        require(stage)
    expected = ["distance_definitions.json", "latent_cache_manifest.csv", "train_distance_stats.csv",
                "val_distance_scores.csv", "val_distance_selection.csv", "selected_distances.json",
                "holdout_distance_scores.csv", "holdout_distance_correlations.csv"]
    assert all((S1 / name).is_file() for name in expected)
    for file, digest in p["manifest_hashes"].items():
        assert sha(file) == digest
    stats = read_scores(S2 / "validation_normalization_stats.csv")
    raw_val = read_scores(S1 / "val_distance_scores.csv")
    val = read_scores(S2 / "val_normalized_scores.csv")
    hold = read_scores(S4 / "holdout_distance_scores.csv")
    train = read_scores(S1 / "train_distance_scores.csv")
    for frame, split, n in [(train,"train",9040),(val,"val",1860),(hold,"holdout",1625)]:
        assert len(frame) == n and frame.split.eq(split).all()
        assert not frame.duplicated(KEY).any()
        assert not frame.dataset.str.contains("KONIQ", case=False).any()
        manifest = read_scores(S0 / f"{split}_manifest.csv")
        assert set(map(tuple, frame[KEY].to_numpy())) == set(map(tuple, manifest[KEY].to_numpy()))
    for row in stats.itertuples():
        x = raw_val.loc[raw_val.dataset.eq(row.dataset), f"{row.pole}_{row.distance}"].to_numpy(float)
        assert len(x) == row.N
        np.testing.assert_allclose([x.mean(), x.std(ddof=0)], [row.mean, row.std_population], rtol=1e-12)
    expected_hold = normalize(hold, stats)
    max_z_error = 0.0
    for col in [c for c in expected_hold if c.startswith(("z_", "Qz_D"))]:
        error = np.abs(expected_hold[col] - hold[col]).max()
        assert error < 1e-10
        max_z_error = max(max_z_error, error)
    selected = read_json(S3 / "selected_distances.json")
    table = read_scores(S3 / "val_distance_selection.csv")
    for pole in POLES:
        rows = []
        for distance in DISTANCES:
            rhos = {key: corr(val.loc[val.distortion_type.eq(kind), f"{pole}_{distance}"],
                              val.loc[val.distortion_type.eq(kind), "severity_or_level"])["SRCC"]
                    for key, kind in LOCKED.items()}
            score = severity_selection(pole, rhos)
            old = table[table.pole.eq(pole) & table.distance.eq(distance)].iloc[0]
            assert np.isclose(score, old.selection_score, rtol=1e-12)
            if distance in ELIGIBLE:
                rows.append((score, distance))
        winner = sorted(rows, key=lambda r: (-r[0], r[1]))[0][1]
        assert winner == selected[f"{pole}_distance"]
    q = hold[f"z_EA_{selected['EA_distance']}"] - hold[f"z_EH_{selected['EH_distance']}"]
    np.testing.assert_allclose(q, hold.Qz_selected, rtol=1e-12, atol=1e-12)
    mos, severity, pairs = holdout_metrics(hold, arms(selected))
    for computed, name, cols in [
        (mos, "holdout_distance_correlations.csv", ["SRCC", "Pearson"]),
        (severity, "holdout_severity_by_distortion.csv", ["SRCC", "Pearson"]),
        (pairs, "holdout_pair_order_accuracy.csv", ["strict_accuracy", "accuracy_half_credit_ties"])]:
        saved = read_scores(S4 / name)
        assert len(saved) == len(computed)
        np.testing.assert_allclose(computed[cols].to_numpy(float), saved[cols].to_numpy(float),
                                   rtol=1e-11, atol=1e-12)
    start = read_json(S4 / "holdout_started.json")
    end = read_json(S4 / "holdout_completed.json")
    assert selected["frozen_at"] < start["started_at"] < end["completed_at"]
    assert sha(S3 / "selected_distances.json") == start["selection_sha256"] == end["selection_sha256"]
    for cache_manifest_file in [S1 / "latent_cache_manifest.csv", S4 / "holdout_latent_cache_manifest.csv"]:
        cache = read_scores(cache_manifest_file)
        for filename, group in cache.groupby("posterior_file"):
            with np.load(filename, allow_pickle=False) as arr:
                assert arr["mu"].shape == arr["logvar"].shape == (len(group), 100)
                for distance in ("D0", "D1", "D2", "D3"):
                    finite(arr[distance], filename)
                info = read_json(Path(filename).parent / "cache.json")
                assert sha(filename) == info["posterior_sha256"]
                recon = np.load(Path(filename).parent / "reconstruction.npy", mmap_mode="r")
                assert recon.shape == (len(group), 3, 256, 256)
                assert np.isfinite(recon[[0,-1]]).all()
    result = {"verified_at": now(), "complete": True, "all_stages_verified": True,
              "counts": p["counts"], "checkpoint_identity_verified": True,
              "validation_only_selection_and_normalization": True,
              "holdout_gated_after_selection": True, "max_z_formula_error": max_z_error,
              "all_correlations_and_pair_accuracy_recomputed": True,
              "D1_rejected_as_full_posterior_distance": True,
              "selected_distances": {k: selected[f"{k}_distance"] for k in POLES},
              "no_new_training": True, "no_koniq_images_loaded": True}
    write_json(HERE / "verification.json", result)
    log(f"Verification complete: {result['selected_distances']}.")


class Tee:
    def __init__(self, original, stream):
        self.original, self.stream = original, stream

    def write(self, text):
        self.original.write(text); self.stream.write(text); self.stream.flush()

    def flush(self):
        self.original.flush(); self.stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "stage0", "stage1", "stage2", "stage3", "stage4",
                                           "rebuild-holdout-tables", "verify"], default="all")
    args = parser.parse_args()
    for folder in STAGES:
        folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    base.set_seed(SEED)
    device = torch.device("cuda")
    if args.stage in ("all", "stage0", "stage1", "stage3", "stage4") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment; no CPU fallback.")
    with (HERE / "run_log.txt").open("a", buffering=1) as stream:
        with contextlib.redirect_stdout(Tee(sys.stdout, stream)), contextlib.redirect_stderr(Tee(sys.stderr, stream)):
            log(f"Start stage={args.stage}; GPU inference, frozen model evaluation.")
            actions = {"stage0": stage0, "stage1": lambda: stage1(device), "stage2": stage2,
                       "stage3": stage3, "stage4": lambda: stage4(device), "verify": verify}
            if args.stage == "rebuild-holdout-tables":
                stage4(device, rebuild_tables=True)
            else:
                for name in actions if args.stage == "all" else [args.stage]:
                    actions[name]()


if __name__ == "__main__":
    main()
