"""Task 3 diagnostics, reusing Task 2 distances and the historical D0."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "day12_task2"))
from run_task2 import canonical, read_csv
from reference_math import diagonal_distances
from score import sz_from_stats
import numpy as np
import torch

DISTANCES = ("D0", "KL", "W2_squared", "Bhattacharyya")
MARGIN = 0.1
NORM_EPS = 1e-8
GAP_EPS = 1e-6
DATASETS = ("KADID-10k", "TID2013")


def energies(mu, logvar, legacy, aggregate=None):
    values = {"D0": sz_from_stats(mu.float(), logvar.float(),
                                  legacy["mu_ref"], legacy["Sigma_ref"],
                                  eps=1e-8, sigma_t_max=1.0, mu_only=True).double()}
    if aggregate is not None:
        other = diagonal_distances(mu, logvar, aggregate["mu_agg"], aggregate["v_agg"])
        values.update(KL=other["KL_q_to_reference"], W2_squared=other["W2_squared"],
                      Bhattacharyya=other["Bhattacharyya"])
    return values


def gap(mild, severe, pole):
    if pole not in ("EA", "EH"):
        raise ValueError("Unknown pole")
    return mild - severe if pole == "EA" else severe - mild


def normalize(x, mean, std):
    return (x - mean) / (std + NORM_EPS)


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    v = x[finite]
    result = {"n": len(x), "finite_count": int(finite.sum()),
              "nonfinite_count": int((~finite).sum())}
    result.update({k: float(f(v)) if len(v) else None for k, f in
                   (("mean", np.mean), ("std_ddof0", np.std), ("min", np.min), ("max", np.max))})
    result["nonconstant"] = bool(len(v) > 1 and np.ptp(v) > 0)
    return result


def pair_arrays(raw, mean, std, mild, severe, pole):
    z = normalize(np.asarray(raw, dtype=np.float64), mean, std)
    g = gap(z[mild], z[severe], pole)
    return {"z_mild": z[mild], "z_severe": z[severe], "gap": g,
            "strict_correct": g > 0, "exact_tie": z[mild] == z[severe],
            "negligible_gap": np.abs(g) < GAP_EPS,
            "rank_loss": np.maximum(0, MARGIN - g), "hinge_active": g < MARGIN}


def pair_summary(arrays):
    result = {"n_pairs": len(arrays["gap"])}
    for key in ("strict_correct", "exact_tie", "negligible_gap", "hinge_active"):
        result[key + "_count"] = int(arrays[key].sum())
        result[key + "_fraction"] = float(arrays[key].mean())
    for key in ("gap", "rank_loss"):
        for stat, value in summarize(arrays[key]).items():
            result[key + "_" + stat] = value
    for label, q in (("p01", .01), ("p05", .05), ("p25", .25),
                     ("p50", .5), ("p75", .75), ("p95", .95), ("p99", .99)):
        result["gap_" + label] = float(np.quantile(arrays["gap"], q))
    return result


def validation_summary(raw, mean, std):
    raw = np.asarray(raw, dtype=np.float64)
    z = normalize(raw, mean, std)
    result = {"raw_" + k: v for k, v in summarize(raw).items()}
    result.update({"standardized_" + k: v for k, v in summarize(z).items()})
    _, counts = np.unique(z[np.isfinite(z)], return_counts=True)
    ties = int(np.sum(counts * (counts - 1) // 2))
    possible = len(z) * (len(z) - 1) // 2
    result.update(exact_tied_unordered_pairs=ties, all_unordered_pairs=possible,
                  exact_tie_rate=ties / possible if possible else 0.0,
                  duplicate_score_image_fraction=float(np.sum(counts[counts > 1]) / len(z)))
    return result


def loss_coefficients(raw, mean, std, mild, severe, pole):
    """d(mean all-pair hinge loss)/d(raw image scores); stats stay constant."""
    x = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
    z = normalize(x, mean, std)
    loss = torch.relu(MARGIN - gap(z[mild], z[severe], pole)).mean()
    return torch.autograd.grad(loss, x)[0].detach(), loss.item()


def locked_membership(split_rows, pair_rows):
    """Use exact paths and dataset-qualified identities, never basename joins."""
    by_path, identities = {}, set()
    for source in split_rows:
        p = canonical(source["distorted_path"])
        key = (source["dataset"], source["image_id"])
        if p in by_path or key in identities:
            raise ValueError("Duplicate image in authoritative split")
        by_path[p] = source
        identities.add(key)
    paired_paths, seen_pairs, clean_pairs = set(), set(), []
    for i, pair in enumerate(pair_rows):
        endpoints = []
        for side in ("mild", "severe"):
            p = canonical(pair["path_" + side])
            row = by_path.get(p)
            if row is None or row["split"] != "train" or row["dataset"] not in DATASETS:
                raise ValueError(f"Forbidden or unmatched pair image: {p}")
            for key in ("dataset", "ref_id", "distortion_type"):
                if row[key] != pair[key]:
                    raise ValueError(f"Pair {key} mismatch: {p}")
            if row["image_id"] != pair["image_id_" + side]:
                raise ValueError("Pair image identity mismatch")
            if float(row["severity_or_level"]) != float(pair["sev_" + side]):
                raise ValueError("Pair severity mismatch")
            if p == canonical(row["ref_path"]) or float(row["severity_or_level"]) <= 0:
                raise ValueError("Pristine row in severity pair")
            paired_paths.add(p)
            endpoints.append(p)
        if float(pair["sev_mild"]) >= float(pair["sev_severe"]):
            raise ValueError("Pair is not mild < severe")
        if tuple(endpoints) in seen_pairs:
            raise ValueError("Duplicate pair")
        seen_pairs.add(tuple(endpoints))
        clean_pairs.append({"pair_index": i, **{k: pair[k] for k in (
            "dataset", "ref_id", "distortion_type", "image_id_mild", "image_id_severe",
            "sev_mild", "sev_severe")}, "path_mild": endpoints[0], "path_severe": endpoints[1]})
    train, val = [], []
    groups = defaultdict(list)
    for p, r in by_path.items():
        if r["dataset"] not in DATASETS or r["split"] not in ("train", "val"):
            continue
        clean = {k: r[k] for k in ("dataset", "image_id", "ref_id", "distortion_type", "split")}
        clean.update(path=p, severity=float(r["severity_or_level"]))
        if r["split"] == "train":
            groups[(r["dataset"], r["ref_id"], r["distortion_type"])].append(clean)
            if p in paired_paths:
                train.append(clean)
        else:
            val.append(clean)
    expected = set()
    for members in groups.values():
        for a in members:
            for b in members:
                if a["severity"] < b["severity"]:
                    expected.add((a["path"], b["path"]))
    if expected != seen_pairs:
        raise ValueError("Pair CSV differs from the locked all-severity-pairs rule")
    if len(paired_paths) != len(train):
        raise ValueError("Unique training image mismatch")
    ref_splits = defaultdict(set)
    for r in split_rows:
        if r["dataset"] in DATASETS:
            ref_splits[(r["dataset"], r["ref_id"])].add(r["split"])
    if any(len(splits) != 1 for splits in ref_splits.values()):
        raise ValueError("Reference identity overlaps train/val/holdout")
    indices = {r["path"]: i for i, r in enumerate(train)}
    a = np.array([indices[p["path_mild"]] for p in clean_pairs])
    b = np.array([indices[p["path_severe"]] for p in clean_pairs])
    return train, val, clean_pairs, a, b
