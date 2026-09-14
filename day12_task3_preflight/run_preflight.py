#!/usr/bin/env python3
"""Frozen Task 3 preflight. Autograd is diagnostic; no optimizer is used."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "day12_task2"))
from run_task2 import (EA_CHECKPOINT, EH_CHECKPOINT, EH_ANCESTOR, SPLIT_MANIFEST,
                       ReferenceDataset, posterior, load_checkpoint, digest,
                       json_digest, read_csv, write_json, write_csv)
from external.model import CVAEGenerator_v2
from preflight_core import (DISTANCES, MARGIN, NORM_EPS, GAP_EPS, DATASETS,
                            energies, normalize, summarize, pair_arrays, pair_summary,
                            validation_summary, loss_coefficients, locked_membership)
import numpy as np
import torch
from torch.utils.data import DataLoader

PAIRS = ROOT / "day11_day12/pairs_severity_train.csv"
TASK2 = ROOT / "day12_task2/reference_stats"
LOCKED_VAL = ROOT / "day12_task1_distance_ablation/stage0_controls/val_manifest.csv"
PDF = Path("/home/projectwork/Downloads/day12_Task3.pdf")
LOG = logging.getLogger("task3")
BLOCK = "BLOCKED_MISSING_ORIGINAL_EH_REFERENCE_MANIFEST"


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(path, rows, fields)


def protocol(batch_size):
    return {
        "task": "Day 12 Task 3", "source_pdf": str(PDF), "source_pdf_sha256": digest(PDF),
        "training_performed": False, "optimizer_steps": 0,
        "weight_updates": 0, "ranked_checkpoints_created": 0,
        "lambda_selection_performed": False, "holdout_scoring_performed": False,
        "MOS_used": False, "KonIQ_images_opened": 0,
        "EA_reference": "Exact verified Task 2 aggregate; historical KonIQ component retained",
        "D0_reference": "Original stored tensors, unchanged for each pre-ranking checkpoint",
        "aggregate_formulas": "Imported from day12_task2/reference_math.py without modification",
        "KL_direction": "image posterior || aggregate reference", "W2": "squared",
        "variance_clamp_min": 1e-8, "variance_upper_cap": None,
        "normalization": "(raw - pooled unique-train mean) / (population train std + 1e-8)",
        "ddof": 0, "normalization_epsilon": NORM_EPS,
        "normalization_fit_split": "severity-pair training unique images only",
        "validation_normalization": "Apply frozen training statistics; do not fit on validation",
        "EA_gap": "normalized mild - normalized severe",
        "EH_gap": "normalized severe - normalized mild", "margin": MARGIN,
        "strict_pair_accuracy": "fraction gap > 0; ties receive no credit",
        "exact_tie": "float64 standardized mild == standardized severe; no tolerance",
        "negligible_gap": "abs(standardized gap) < 1e-6", "negligible_threshold": GAP_EPS,
        "validation_tie_rate": "equal-score unordered image pairs / all unordered image pairs",
        "gradient_objective": "mean hinge loss over every locked severity-training pair",
        "gradient_method": "exact chain-rule accumulation through every unique training image",
        "gradient_norm_mean_max_scope": "fixed image-batch contributions to full mean-loss gradient",
        "gradient_batch_size": batch_size,
        "per_pair_gradient_scope": "posterior mu/logvar leaf gradients of individual hinge loss",
        "inactive_pair_zero_gradient": "expected", "D0_logvar_zero_gradient": "expected",
        "reference_and_normalization_gradients": "detached constants",
        "precision": "float32 encoder, historical float32 D0, float64 other distances and normalization",
        "preprocess": "Task 2 RGB, upscale short side only if <256, center crop 256, divide by 255",
        "model_mode": "eval; no decoder, augmentation, autocast, optimizer, or state updates",
        "pair_rule": "original Day 11 all mild<severe pairs, same dataset/ref_id/type",
        "EH_D0_partial_control": "Legacy D0 can execute independently of missing aggregate reference",
    }


def audit_inputs(out):
    train, val, pairs, a, b = locked_membership(read_csv(SPLIT_MANIFEST), read_csv(PAIRS))
    locked_val = read_csv(LOCKED_VAL)
    actual = {(r["dataset"], r["image_id"]) for r in val}
    if actual != {(r["dataset"], r["image_id"]) for r in locked_val}:
        raise ValueError("Validation membership differs from the previously locked validation file")
    if (len(train), len(val), len(pairs)) != (9040, 1860, 18080):
        raise ValueError("Locked input counts changed; review before execution")
    for rows in (train, val):
        for row in rows:
            path = Path(row["path"])
            row["image_sha256"] = digest(path)
            row["bytes"] = path.stat().st_size
    train_hashes = {r["image_sha256"] for r in train}
    if train_hashes & {r["image_sha256"] for r in val}:
        raise ValueError("Identical image bytes cross train/validation")
    table(out / "inputs/train_images.csv", train)
    table(out / "inputs/validation_images.csv", val)
    table(out / "inputs/pairs_used.csv", pairs)
    audit = {
        "status": "PASS", "unique_training_images": len(train), "training_pairs": len(pairs),
        "validation_images": len(val), "training_composition": dict(Counter(r["dataset"] for r in train)),
        "validation_composition": dict(Counter(r["dataset"] for r in val)),
        "train_val_path_overlap": 0, "train_val_byte_overlap": 0,
        "reference_identity_split_overlap": 0, "pristine_pair_images": 0,
        "KonIQ_scoring_images": 0, "MOS_and_abs_dEH_removed_from_task_pair_copy": True,
        "all_severity_pairs_exactly_match": True, "all_validation_images_exactly_match": True,
        "holdout_images_opened_or_scored": 0,
        "holdout_note": "Split metadata is read for exclusion only; no holdout image content or scores read",
    }
    write_json(out / "inputs/split_audit.json", audit)
    return train, val, pairs, a, b, audit


def references(out):
    ea = json.loads((TASK2 / "EA/manifest.json").read_text())
    verification = json.loads((TASK2 / "verification.json").read_text())
    if ea["status"] != "COMPLETE" or verification["independent_aggregate_recomputation"] != "PASS":
        raise ValueError("Task 2 EA aggregate has not passed verification")
    for path, sha in ea["source_hashes"].items():
        if digest(path) != sha:
            raise ValueError(f"Task 2 verified source changed: {path}")
    for name, key in (("aggregate_reference.npz", "aggregate_sha256"),
                      ("legacy_reference.npz", "legacy_sha256"),
                      ("reference_posteriors.npz", "posterior_sha256"),
                      ("reference_images.csv", "image_manifest_sha256")):
        if digest(TASK2 / "EA" / name) != ea[key]:
            raise ValueError(f"Task 2 EA artifact checksum mismatch: {name}")
    with np.load(TASK2 / "EA/aggregate_reference.npz", allow_pickle=False) as data:
        aggregate = {k: torch.from_numpy(data[k].copy()) for k in ("mu_agg", "v_agg")}
        np.testing.assert_array_equal(data["v_aggregate_raw"], data["v_between"] + data["v_within"])
        if int(data["ddof"]) != 0 or int(data["n_images"]) != 2000:
            raise ValueError("Unexpected Task 2 reference statistics")
    for t in aggregate.values():
        if t.shape != (100,) or not torch.isfinite(t).all():
            raise ValueError("Invalid aggregate reference")
    if not torch.all(aggregate["v_agg"] >= 1e-8):
        raise ValueError("Nonpositive reference variance")
    ckpts = {"EA": load_checkpoint(EA_CHECKPOINT, 5), "EH": load_checkpoint(EH_CHECKPOINT, 20)}
    for pole, ck in ckpts.items():
        with np.load(TASK2 / pole / "legacy_reference.npz", allow_pickle=False) as saved:
            for key in ("mu_ref", "Sigma_ref"):
                original = ck[key].numpy()
                if original.dtype != saved[key].dtype or not np.array_equal(original, saved[key]):
                    raise ValueError(f"{pole} original legacy reference mismatch")
        shutil.copy2(TASK2 / pole / "legacy_reference.npz", out / pole / "legacy_reference.npz")
    write_json(out / "EA/reference_reuse.json", {
        "status": "PASS", "source": str(TASK2 / "EA/aggregate_reference.npz"),
        "aggregate_sha256": ea["aggregate_sha256"], "legacy_sha256": ea["legacy_sha256"],
        "checkpoint_sha256": ea["source_hashes"][str(EA_CHECKPOINT)],
        "reference_refitted_in_task3": False, "Task2_formula_source_unchanged": True})
    return ckpts, aggregate


def audit_eh(out, ck):
    prior = json.loads((TASK2 / "EH/manifest.json").read_text())
    if digest(EH_CHECKPOINT) != prior["checkpoint_sha256"]:
        raise ValueError("EH pre-ranking checkpoint changed")
    ancestor = torch.load(EH_ANCESTOR, map_location="cpu", weights_only=False)
    if digest(EH_ANCESTOR) != prior["ancestor_sha256"]:
        raise ValueError("EH ancestor checkpoint changed")
    equality = {k: torch.equal(ck[k], ancestor[k]) for k in ("mu_ref", "Sigma_ref")}
    if not all(equality.values()):
        raise ValueError("EH reference lineage does not match recorded evidence")
    candidates = [Path(prior["ancestor_training_manifest"]), Path(prior["ancestor_saved_training_list"])]
    staging = Path("/home/projectwork/wacv_push_staging_20260904_232208")
    candidates += [staging / p.relative_to(ROOT) for p in candidates.copy()]
    locations = [{"path": str(p), "exists": p.is_file()} for p in candidates]
    if any(r["exists"] for r in locations):
        raise ValueError("An original-list candidate is now present. Audit its exact sampling provenance before proceeding")
    record = {
        **prior, "status": BLOCK, "task3_audit_utc": datetime.now(timezone.utc).isoformat(),
        "original_reference_manifest_recovered": False, "recovery_candidates": locations,
        "original_reference_count": None, "reference_manifest_sha256": None,
        "reference_image_hashes": None, "reference_dataset_composition": None,
        "aggregate_reference_created": False, "exact_reference_preprocessing_verified": False,
        "legacy_tensor_equality_with_ancestor": equality,
        "reference_count_not_inferred_from_later_loader": True,
        "current_encoder_preprocess": protocol(16)["preprocess"],
        "evidence": [
            "Stored legacy tensors exactly match checkpoints/best.pth",
            "Original train_pristine.txt and original run train_files.txt are missing",
            "Later hr_combined_ft1 log says restored frozen reference, not fitted its candidate pool",
            "A deterministic new sample of the later pool cannot establish the original image identities"],
        "required_to_unblock": ["Exact original reference list, or original ordered source list",
                                "Historical reference sampling rule and seed",
                                "Historical preprocessing and original image files",
                                "Verifiable validation/holdout exclusion"],
        "EH_D0_executable": True, "EH_KL_W2_Bhattacharyya_executable": False,
    }
    write_json(out / "EH_reference_manifest.json", record)
    write_json(out / "EH/recovery_audit.json", record)
    comparison = []
    for key in ("mu_ref", "Sigma_ref"):
        original = ck[key].numpy()
        comparison.append({"tensor": key, "status": "LEGACY_PRESERVED_AGGREGATE_BLOCKED",
                           "legacy_dtype": str(original.dtype), "legacy_shape": str(original.shape),
                           "legacy_min": float(original.min()), "legacy_max": float(original.max()),
                           "legacy_mean": float(original.mean()), "legacy_export_max_abs_difference": 0.0,
                           "aggregate_mean": None, "aggregate_max_abs_difference": None,
                           "v_between_mean": None, "v_within_mean": None, "reason": BLOCK})
    table(out / "EH_reference_comparison.csv", comparison)
    return record


def model_for(ck, device):
    cfg = ck["config"]
    if int(cfg["img"]) != 256 or int(cfg["ldim"]) != 100:
        raise ValueError("Unsupported checkpoint dimensions")
    model = CVAEGenerator_v2(latent_dim=100, image_size=256)
    model.load_state_dict(ck["generator"], strict=True)
    return model.to(device).eval()


def extract(model, ck, agg, rows, pole, split, out, args, fingerprint):
    path = out / pole / f"{split}_posteriors.npz"
    meta = out / pole / f"{split}_cache.json"
    if path.exists() and meta.exists():
        previous = json.loads(meta.read_text())
        if previous["fingerprint"] == fingerprint and previous["sha256"] == digest(path):
            LOG.info("%s %s: reusing verified Task 3 cache", pole, split)
            with np.load(path, allow_pickle=False) as data:
                if list(data["path"]) != [r["path"] for r in rows]:
                    raise ValueError("Cache image order mismatch")
                return data["mu"].copy(), data["logvar"].copy(), {
                    d: data["score_" + d].copy() for d in DISTANCES if "score_" + d in data}
    loader = DataLoader(ReferenceDataset(rows, 256), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False, pin_memory=args.device.startswith("cuda"))
    ms, ls, vs = [], [], {d: [] for d in (DISTANCES if agg is not None else ("D0",))}
    with torch.no_grad():
        for i, x in enumerate(loader):
            mu, lv = posterior(model, x.to(args.device, non_blocking=True))
            if not torch.isfinite(mu).all() or not torch.isfinite(lv).all():
                raise ValueError("Nonfinite encoder posterior")
            ms.append(mu.cpu().numpy()); ls.append(lv.cpu().numpy())
            for name, value in energies(mu, lv, ck, agg).items():
                vs[name].append(value.cpu().numpy())
            if i % 100 == 0 or i + 1 == len(loader):
                LOG.info("%s %s scored %d/%d", pole, split, min((i+1)*args.batch_size, len(rows)), len(rows))
    mu, lv = np.concatenate(ms), np.concatenate(ls)
    scores = {d: np.concatenate(v) for d, v in vs.items()}
    np.savez_compressed(path, mu=mu, logvar=lv, path=np.array([r["path"] for r in rows]),
                        **{"score_" + k: v for k, v in scores.items()})
    write_json(meta, {"fingerprint": fingerprint, "sha256": digest(path), "n_images": len(rows),
                      "checkpoint_epoch": int(ck["epoch"]), "pole": pole, "split": split,
                      "batch_size": args.batch_size, "device": args.device, "AMP": False})
    return mu, lv, scores


def posterior_pair_gradients(mu, lv, ck, agg, mild, severe, stats, arrays, pole, device):
    results = {}
    for d in arrays:
        m = torch.tensor(np.stack([mu[mild], mu[severe]], axis=1).reshape(-1, mu.shape[1]),
                         device=device, requires_grad=True)
        l = torch.tensor(np.stack([lv[mild], lv[severe]], axis=1).reshape(-1, lv.shape[1]),
                         device=device, requires_grad=True)
        v = energies(m, l, ck, agg)[d].reshape(-1, 2)
        z = normalize(v, stats[d]["mean"], stats[d]["std_ddof0"])
        g = z[:, 0] - z[:, 1] if pole == "EA" else z[:, 1] - z[:, 0]
        loss = torch.relu(MARGIN - g)
        if not np.array_equal((g.detach().cpu().numpy() < MARGIN), arrays[d]["hinge_active"]):
            raise ValueError("Cached and posterior-recomputed hinge active sets differ")
        dm, dl = torch.autograd.grad(loss.sum(), (m, l), allow_unused=True)
        dl = torch.zeros_like(l) if dl is None else dl
        dm = dm.double().reshape(-1, 2 * mu.shape[1])
        dl = dl.double().reshape(-1, 2 * mu.shape[1])
        norms_m, norms_l = dm.norm(dim=1), dl.norm(dim=1)
        results[d] = {
            "posterior_mu_grad_l2": norms_m.detach().cpu().numpy(),
            "posterior_logvar_grad_l2": norms_l.detach().cpu().numpy(),
            "posterior_pair_loss_grad_l2": (norms_m.square() + norms_l.square()).sqrt().detach().cpu().numpy(),
            "posterior_grad_finite": (torch.isfinite(dm).all(1) & torch.isfinite(dl).all(1)).cpu().numpy(),
        }
    return results


def encoder_gradients(model, ck, agg, rows, raw, stats, a, b, pole, out, args):
    coeff, objectives = {}, {}
    for d, scores in raw.items():
        c, objective = loss_coefficients(scores, stats[d]["mean"], stats[d]["std_ddof0"], a, b, pole)
        coeff[d], objectives[d] = c.to(args.device), objective
    targets = {"enc1.weight": model.enc1.weight, "fc_mu.weight": model.fc_mu.weight,
               "fc_log_var.weight": model.fc_log_var.weight}
    total = {d: {p: torch.zeros_like(v, dtype=torch.float64, device="cpu")
                 for p, v in targets.items()} for d in raw}
    norms = {d: {p: [] for p in targets} for d in raw}
    batch_rows, replay = [], {d: [] for d in raw}
    loader = DataLoader(ReferenceDataset(rows, 256), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
    offset = 0
    for i, x in enumerate(loader):
        mu, lv = posterior(model, x.to(args.device, non_blocking=True))
        scores = energies(mu, lv, ck, agg)
        n = len(x)
        for j, (d, value) in enumerate(scores.items()):
            replay[d].append(value.detach().cpu().numpy())
            grads = torch.autograd.grad(value, tuple(targets.values()),
                                        grad_outputs=coeff[d][offset:offset+n],
                                        retain_graph=j + 1 < len(scores), allow_unused=True)
            for (p, param), grad in zip(targets.items(), grads):
                unused = grad is None
                g = torch.zeros_like(param, device="cpu", dtype=torch.float64) if unused else grad.detach().double().cpu()
                norm = g.norm().item()
                finite = bool(torch.isfinite(g).all())
                if not finite:
                    raise ValueError(f"Nonfinite ranking-path gradient: {pole}/{d}/{p}")
                total[d][p] += g
                norms[d][p].append(norm)
                batch_rows.append({"pole": pole, "distance": d, "parameter": p,
                                   "image_batch": i, "first_image": offset, "n_images": n,
                                   "contribution_gradient_l2": norm, "finite": finite,
                                   "unused_graph_path": unused})
        offset += n
        if i % 100 == 0 or i + 1 == len(loader):
            LOG.info("%s full-pair-loss gradient: %d/%d unique images", pole, offset, len(rows))
    if offset != len(rows):
        raise ValueError("Incomplete gradient pass")
    checks, summary, grad_arrays = {}, [], {}
    for d in raw:
        repeated = np.concatenate(replay[d])
        max_error = float(np.max(np.abs(repeated - raw[d])))
        # Same batch layout, eval mode, deterministic encoder; replays must be exact.
        if not np.array_equal(repeated, raw[d]):
            raise ValueError(f"Scoring/gradient replay mismatch for {pole}/{d}: {max_error}")
        checks[d] = {"exact_score_replay": True, "maximum_absolute_difference": max_error}
        for p in targets:
            g = total[d][p]
            full_norm = g.norm().item()
            expected_zero = d == "D0" and p == "fc_log_var.weight"
            passed = bool(torch.isfinite(g).all()) and (full_norm == 0 if expected_zero else full_norm > 0)
            summary.append({"pole": pole, "distance": d, "status": "PASS" if passed else "FAIL",
                            "parameter": p, "n_unique_training_images": len(rows), "n_pairs": len(a),
                            "mean_preflight_ranking_loss": objectives[d],
                            "full_mean_loss_gradient_l2": full_norm,
                            "mean_gradient_norm": float(np.mean(norms[d][p])),
                            "max_gradient_norm": float(np.max(norms[d][p])),
                            "min_gradient_norm": float(np.min(norms[d][p])),
                            "gradient_norm_scope": "fixed image-batch contribution to full mean loss",
                            "n_image_batches": len(norms[d][p]), "image_batch_size": args.batch_size,
                            "all_gradients_finite": True, "full_gradient_nonzero": full_norm > 0,
                            "zero_gradient_expected": expected_zero,
                            "scoring_gradient_replay_exact": True})
            grad_arrays[d + "__" + p.replace(".", "_")] = g.numpy()
    table(out / pole / "gradient_batch_contributions.csv", batch_rows)
    np.savez_compressed(out / pole / "full_mean_loss_gradients.npz", **grad_arrays)
    write_json(out / pole / "gradient_score_replay.json", checks)
    return summary


def run_pole(pole, ck, agg, train, val, pairs, a, b, out, args, fingerprint):
    model = model_for(ck, args.device)
    mu, lv, raw = extract(model, ck, agg, train, pole, "train", out, args, fingerprint)
    stats = {d: summarize(v) for d, v in raw.items()}
    if any(s["nonfinite_count"] for s in stats.values()):
        table(out / f"{pole}_training_distance_stats.csv", [{"distance": d, **s} for d, s in stats.items()])
        raise ValueError(f"{pole} nonfinite training distances")
    write_json(out / pole / "frozen_training_normalization.json", {
        "fit_split": "unique severity-training images", "datasets": list(DATASETS),
        "ddof": 0, "epsilon": NORM_EPS, "n_unique_images": len(train), "distances": stats})
    arrays = {d: pair_arrays(v, stats[d]["mean"], stats[d]["std_ddof0"], a, b, pole) for d, v in raw.items()}
    leaf = posterior_pair_gradients(mu, lv, ck, agg, a, b, stats, arrays, pole, args.device)
    gradients = encoder_gradients(model, ck, agg, train, raw, stats, a, b, pole, out, args)
    train_rows, pair_rows = [], []
    for d in raw:
        train_rows.append({"pole": pole, "distance": d, "status": "COMPLETE",
                           "n_unique_images": len(train), "n_pairs": len(pairs),
                           **stats[d], "normalization_epsilon": NORM_EPS, "ddof": 0})
        summary = pair_summary(arrays[d])
        grad = leaf[d]["posterior_pair_loss_grad_l2"]
        pair_rows.append({"pole": pole, "distance": d, "status": "COMPLETE",
                          "n_unique_images": len(train), **summary,
                          "posterior_pair_gradient_mean_l2": float(grad.mean()),
                          "posterior_pair_gradient_max_l2": float(grad.max()),
                          "posterior_gradients_all_finite": bool(leaf[d]["posterior_grad_finite"].all()),
                          "active_pairs_with_zero_posterior_gradient": int(((grad == 0) & arrays[d]["hinge_active"]).sum()),
                          "inactive_pairs_with_nonzero_posterior_gradient": int(((grad != 0) & ~arrays[d]["hinge_active"]).sum()),
                          "gap_polarity": "mild_minus_severe" if pole == "EA" else "severe_minus_mild"})
        detail = [{**p, "pole": pole, "distance": d,
                   **{k: v[i].item() for k, v in arrays[d].items()},
                   **{k: v[i].item() for k, v in leaf[d].items()}} for i, p in enumerate(pairs)]
        table(out / pole / f"pair_details_{d}.csv", detail)
    # Freeze training statistics above before opening any validation images.
    _, _, val_raw = extract(model, ck, agg, val, pole, "val", out, args, fingerprint)
    val_rows = []
    for d, values in val_raw.items():
        for dataset in ("KADID+TID", *DATASETS):
            selected = np.array([dataset == "KADID+TID" or r["dataset"] == dataset for r in val])
            val_rows.append({"pole": pole, "distance": d, "dataset": dataset, "status": "COMPLETE",
                             "n_images": int(selected.sum()),
                             "normalization_train_mean": stats[d]["mean"],
                             "normalization_train_std_ddof0": stats[d]["std_ddof0"],
                             "normalization_epsilon": NORM_EPS,
                             **validation_summary(values[selected], stats[d]["mean"], stats[d]["std_ddof0"])})
    for split, rows, values in (("train", train, raw), ("val", val, val_raw)):
        result = []
        for i, r in enumerate(rows):
            entry = dict(r)
            for d, scores in values.items():
                entry[d + "_raw"] = float(scores[i])
                entry[d + "_standardized"] = float(normalize(scores[i], stats[d]["mean"], stats[d]["std_ddof0"]))
            result.append(entry)
        table(out / pole / f"{split}_scores.csv", result)
    if pole == "EH" and agg is None:
        for d in DISTANCES[1:]:
            for rows in (train_rows, pair_rows, gradients, val_rows):
                rows.append({"pole": "EH", "distance": d, "status": BLOCK,
                             "reason": "Original pristine-reference image pool unavailable; no aggregate substituted"})
    for suffix, rows in (("training_distance_stats", train_rows), ("pair_gap_diagnostics", pair_rows),
                         ("gradient_preflight", gradients), ("validation_preflight", val_rows)):
        table(out / f"{pole}_{suffix}.csv", rows)
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), ck["generator"][name]):
            raise ValueError(f"Model state changed during preflight: {pole}/{name}")
    ready = (all(r["status"] == "PASS" for r in gradients if r.get("parameter"))
             and all(s["nonconstant"] and not s["nonfinite_count"] for s in stats.values())
             and all(not r.get("raw_nonfinite_count", 0) and r.get("raw_nonconstant", True) for r in val_rows)
             and all(r["posterior_gradients_all_finite"] and r["active_pairs_with_zero_posterior_gradient"] == 0
                     and r["inactive_pairs_with_nonzero_posterior_gradient"] == 0
                     for r in pair_rows if r["status"] == "COMPLETE"))
    result = {"available_distances": list(raw), "available_checks_passed": ready,
              "encoder_weights_unchanged": True, "complete": agg is not None,
              "gradient_checks_passed": sum(r["status"] == "PASS" for r in gradients),
              "status": ("PASS" if ready else "FAIL") if agg is not None else "D0_CHECKED_AGGREGATES_BLOCKED"}
    write_json(out / pole / "status.json", result)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size or workers")
    out = args.output.resolve()
    if (out / "preflight_status.json").exists() and not args.rerun:
        parser.error("Outputs already exist. Verify them, or explicitly pass --rerun to replace only this task's outputs")
    for name in ("inputs", "EA", "EH", "verification"):
        (out / name).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback. Run this preflight on a GPU host")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(42)
    write_json(out / "inputs/protocol.json", protocol(args.batch_size))
    shutil.copy2(PDF, out / "inputs/day12_Task3.pdf")
    source_paths = [PAIRS, SPLIT_MANIFEST, LOCKED_VAL, EA_CHECKPOINT, EH_CHECKPOINT, EH_ANCESTOR,
                    TASK2 / "EA/manifest.json", TASK2 / "EA/aggregate_reference.npz",
                    TASK2 / "EH/manifest.json", TASK2 / "verification.json",
                    ROOT / "day12_task2/reference_math.py", ROOT / "day12_task2/run_task2.py",
                    ROOT / "score.py", ROOT / "external/model.py", ROOT / "external/dataloader.py",
                    HERE / "run_preflight.py", HERE / "preflight_core.py", HERE / "README.md", PDF]
    before = {str(p): digest(p) for p in source_paths}
    write_json(out / "inputs/source_hashes.json", before)
    write_json(out / "preflight_status.json", {"status": "RUNNING", "task_complete": False,
                                               "training_performed": False, "holdout_scoring_performed": False})
    try:
        LOG.info("Task 3: frozen preflight only; no training and no holdout scoring")
        ckpts, aggregate = references(out)
        train, val, pairs, a, b, audit = audit_inputs(out)
        fingerprint = json_digest({"source_hashes": before, "protocol": protocol(args.batch_size),
                                   "train": train, "validation": val, "torch": torch.__version__,
                                   "numpy": np.__version__, "device": args.device})
        write_json(out / "inputs/run_identity.json", {"fingerprint": fingerprint,
                    "torch": torch.__version__, "numpy": np.__version__, "device": args.device,
                    "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None})
        LOG.info("Input audit passed: %d train, %d pairs, %d validation", len(train), len(pairs), len(val))
        ea = run_pole("EA", ckpts["EA"], aggregate, train, val, pairs, a, b, out, args, fingerprint)
        eh_reference = audit_eh(out, ckpts["EH"])
        eh = run_pole("EH", ckpts["EH"], None, train, val, pairs, a, b, out, args, fingerprint)
        after = {str(p): digest(p) for p in source_paths}
        if before != after:
            raise ValueError("Read-only research inputs changed during the run")
        write_json(out / "verification/source_integrity.json", {"status": "PASS", "before": before, "after": after})
        status = {"status": "PARTIAL_EH_REFERENCE_BLOCKED", "task_complete": False,
                  "ready_for_matched_distance_training": False,
                  "EA": ea, "EH": eh, "EH_reference": eh_reference["status"], "input_audit": audit,
                  "training_performed": False, "optimizer_steps": 0, "model_weights_unchanged": True,
                  "ranked_checkpoints_created": 0, "lambda_selected": False, "holdout_images_scored": 0,
                  "original_research_inputs_unchanged": True, "completed_utc": datetime.now(timezone.utc).isoformat()}
        write_json(out / "preflight_status.json", status)
        report = ["DAY 12 TASK 3 PREFLIGHT", "STATUS: PARTIAL; original EH reference manifest remains missing.",
                  "", "No training, optimizer, weight update, checkpoint creation, lambda selection, or holdout evaluation.",
                  f"Inputs: {len(train)} unique training images; {len(pairs)} pairs; {len(val)} locked validation images.",
                  "EA: all four distances scored using the pre-ranking checkpoint and verified Task 2 reference.",
                  f"EA available numerical/gradient checks passed: {ea['available_checks_passed']}.",
                  "EH: historical D0 scored with unchanged legacy tensors; aggregate-dependent distances BLOCKED.",
                  f"EH D0 numerical/gradient checks passed: {eh['available_checks_passed']}.",
                  "", "NORMALIZATION AND POLARITY", "Training mean and population std over unique images only.",
                  "Validation uses these frozen training statistics. No validation normalization fit.",
                  "EA gap = mild - severe; EH gap = severe - mild; loss = max(0, 0.1 - standardized gap).",
                  "Exact ties use ==; negligible gaps use abs(gap) < 1e-6, counted separately.",
                  "", "GRADIENT INTERPRETATION", "Full mean-loss gradients use every training pair, through every unique image.",
                  "Mean/max gradient norms describe fixed image-batch contributions, not per-pair encoder gradients.",
                  "Additional per-pair posterior-gradient diagnostics are saved and explicitly labeled.",
                  "Inactive hinge pairs may have zero gradients. D0's zero log-variance gradient is expected.",
                  "Scoring and gradient passes reproduce the exact same raw scores; model state is unchanged.",
                  "", "EH BLOCKER", *eh_reference["evidence"],
                  "Need the exact original reference list, sampling provenance, preprocessing, and original images.",
                  "No new DIV2K/Flickr2K or other replacement pool has been sampled.",
                  "Blocked CSV rows have empty unavailable metrics; they are not zero-valued results.",
                  "", "INTERPRETATION", "This checks numerical readiness, not improved image-quality performance.",
                  "Pair accuracy before training is diagnostic; no distance or lambda is selected here.",
                  "The whole matched-distance experiment is NOT ready while EH aggregate checks remain blocked."]
        for pole in ("EA", "EH"):
            report += ["", pole + " TRAINING PAIR SUMMARY"]
            for row in read_csv(out / f"{pole}_pair_gap_diagnostics.csv"):
                if row["status"] == "COMPLETE":
                    report.append(f"{row['distance']}: strict accuracy={float(row['strict_correct_fraction']):.6f}; "
                                  f"ties={float(row['exact_tie_fraction']):.6f}; "
                                  f"negligible={float(row['negligible_gap_fraction']):.6f}; "
                                  f"mean rank loss={float(row['rank_loss_mean']):.6f}")
                else:
                    report.append(row["distance"] + ": BLOCKED (no aggregate reference)")
        (out / "preflight_report.txt").write_text("\n".join(report) + "\n")
        LOG.info("EA complete; EH D0 complete; EH aggregate preflight BLOCKED. No training performed.")
        return 2
    except Exception as exc:
        write_json(out / "preflight_status.json", {"status": "FAILED", "task_complete": False,
                    "error": repr(exc), "training_performed": False, "holdout_scoring_performed": False})
        LOG.exception("Preflight stopped; do not interpret incomplete outputs as passed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
