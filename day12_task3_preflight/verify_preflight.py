#!/usr/bin/env python3
"""Independent NumPy audit of Task 3 outputs; no training or holdout access."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def json_file(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, rtol=1e-12, atol=1e-12):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=rtol, atol=atol)


def column(data, key):
    return np.array([float(r[key]) for r in data], dtype=np.float64)


def independent_values(mu, lv, legacy, aggregate):
    m32 = mu.astype(np.float32)
    delta32 = m32 - legacy["mu_ref"]
    d0 = np.sqrt(np.sum(delta32 ** 2 / (legacy["Sigma_ref"] + np.float32(1e-8)), axis=1))
    values = {"D0": d0.astype(np.float64)}
    derivatives = {"D0": (delta32.astype(float) / (legacy["Sigma_ref"] + np.float32(1e-8)) / d0[:, None],
                           np.zeros_like(mu, dtype=np.float64))}
    if aggregate is not None:
        mu, lv = mu.astype(float), lv.astype(float)
        vx_raw = np.exp(lv)
        vx = np.maximum(vx_raw, 1e-8)
        vr = np.maximum(aggregate["v_agg"], 1e-8)
        delta = mu - aggregate["mu_agg"]
        bar = (vx + vr) / 2
        values["KL"] = .5 * np.sum(np.log(vr / vx) + vx / vr + delta ** 2 / vr - 1, axis=1)
        values["W2_squared"] = np.sum(delta ** 2 + (np.sqrt(vx) - np.sqrt(vr)) ** 2, axis=1)
        values["Bhattacharyya"] = np.sum(delta ** 2 / bar, axis=1) / 8 + np.sum(np.log(bar / np.sqrt(vx * vr)), axis=1) / 2
        active_clamp = vx_raw >= 1e-8
        derivatives["KL"] = (delta / vr, .5 * (vx / vr - 1) * active_clamp)
        derivatives["W2_squared"] = (2 * delta, (vx - np.sqrt(vx * vr)) * active_clamp)
        derivatives["Bhattacharyya"] = (delta / (4 * bar),
            (-delta ** 2 * vx / (16 * bar ** 2) + vx / (4 * bar) - .25) * active_clamp)
    return values, derivatives


def check_stats(row, x, prefix=""):
    assert int(row[prefix + "finite_count"]) == len(x)
    assert int(row[prefix + "nonfinite_count"]) == 0
    for key, expected in (("mean", x.mean()), ("std_ddof0", x.std(ddof=0)), ("min", x.min()), ("max", x.max())):
        close(float(row[prefix + key]), expected)
    assert row[prefix + "nonconstant"] == str(bool(np.ptp(x) > 0))


def verify_pole(out, pole, train, val, pairs, a, b):
    legacy = dict(np.load(out / pole / "legacy_reference.npz", allow_pickle=False))
    original_legacy = ROOT / "day12_task2/reference_stats" / pole / "legacy_reference.npz"
    assert sha(original_legacy) == sha(out / pole / "legacy_reference.npz")
    aggregate = None
    if pole == "EA":
        aggregate = dict(np.load(ROOT / "day12_task2/reference_stats/EA/aggregate_reference.npz", allow_pickle=False))
    names = ("D0", "KL", "W2_squared", "Bhattacharyya") if pole == "EA" else ("D0",)
    training_stats = rows(out / f"{pole}_training_distance_stats.csv")
    gap_stats = rows(out / f"{pole}_pair_gap_diagnostics.csv")
    val_stats = rows(out / f"{pole}_validation_preflight.csv")
    grad_stats = rows(out / f"{pole}_gradient_preflight.csv")
    frozen = json_file(out / pole / "frozen_training_normalization.json")
    assert frozen["ddof"] == 0 and frozen["epsilon"] == 1e-8 and frozen["n_unique_images"] == len(train)
    raw_by_split, derivative_train, formula_errors = {}, None, {}
    for split, members in (("train", train), ("val", val)):
        file = out / pole / f"{split}_posteriors.npz"
        metadata = json_file(out / pole / f"{split}_cache.json")
        assert sha(file) == metadata["sha256"]
        assert metadata["fingerprint"] == json_file(out / "inputs/run_identity.json")["fingerprint"]
        scored = rows(out / pole / f"{split}_scores.csv")
        assert len(scored) == len(members)
        with np.load(file, allow_pickle=False) as cache:
            assert cache["mu"].shape == cache["logvar"].shape == (len(members), 100)
            assert list(cache["path"]) == [r["path"] for r in members]
            expected, derivatives = independent_values(cache["mu"], cache["logvar"], legacy, aggregate)
            if split == "train":
                derivative_train = derivatives
            raw_by_split[split] = {}
            for d in names:
                raw = cache["score_" + d].copy()
                assert np.isfinite(raw).all() and np.ptp(raw) > 0
                close(raw, expected[d], rtol=5e-7 if d == "D0" else 1e-10, atol=1e-5 if d == "D0" else 5e-12)
                formula_errors[f"{split}/{d}"] = float(np.max(np.abs(raw - expected[d])))
                np.testing.assert_array_equal(column(scored, d + "_raw"), raw)
                raw_by_split[split][d] = raw
                f = frozen["distances"][d]
                z = (raw - f["mean"]) / (f["std_ddof0"] + 1e-8)
                np.testing.assert_array_equal(column(scored, d + "_standardized"), z)
            for r, m in zip(scored, members):
                assert all(r[k] == m[k] for k in ("dataset", "image_id", "path", "split", "image_sha256"))
    for d in names:
        raw = raw_by_split["train"][d]
        mean, std = raw.mean(), raw.std(ddof=0)
        s = next(r for r in training_stats if r["distance"] == d)
        assert s["status"] == "COMPLETE" and int(s["n_pairs"]) == len(pairs)
        assert int(s["n_unique_images"]) == len(train)
        check_stats(s, raw)
        close(frozen["distances"][d]["mean"], mean)
        close(frozen["distances"][d]["std_ddof0"], std)
        z = (raw - mean) / (std + 1e-8)
        g = z[a] - z[b] if pole == "EA" else z[b] - z[a]
        loss = np.maximum(0, .1 - g)
        flag = {"exact_tie": z[a] == z[b], "negligible_gap": np.abs(g) < 1e-6,
                "strict_correct": g > 0, "hinge_active": g < .1}
        detail = rows(out / pole / f"pair_details_{d}.csv")
        assert len(detail) == len(pairs)
        for i, (row, pair) in enumerate(zip(detail, pairs)):
            assert row["pair_index"] == str(i)
            for k in ("path_mild", "path_severe", "image_id_mild", "image_id_severe", "dataset", "ref_id"):
                assert row[k] == pair[k]
        np.testing.assert_array_equal(column(detail, "gap"), g)
        np.testing.assert_array_equal(column(detail, "rank_loss"), loss)
        summary = next(r for r in gap_stats if r["distance"] == d)
        for key, flags in flag.items():
            assert [r[key] for r in detail] == [str(bool(v)) for v in flags]
            assert int(summary[key + "_count"]) == int(flags.sum())
            close(float(summary[key + "_fraction"]), flags.mean())
        for key, values in (("gap", g), ("rank_loss", loss)):
            check_stats(summary, values, prefix=key + "_")
        for label, quantile in (("p01", .01), ("p05", .05), ("p25", .25), ("p50", .5),
                                ("p75", .75), ("p95", .95), ("p99", .99)):
            close(float(summary["gap_" + label]), np.quantile(g, quantile))
        dm, dl = derivative_train[d]
        factor = flag["hinge_active"].astype(float) / (std + 1e-8)
        nm = np.sqrt(np.sum(dm[a] ** 2 + dm[b] ** 2, axis=1)) * factor
        nl = np.sqrt(np.sum(dl[a] ** 2 + dl[b] ** 2, axis=1)) * factor
        close(column(detail, "posterior_mu_grad_l2"), nm, rtol=3e-5, atol=1e-8)
        close(column(detail, "posterior_logvar_grad_l2"), nl, rtol=3e-5, atol=1e-8)
        norm = column(detail, "posterior_pair_loss_grad_l2")
        close(norm, np.sqrt(nm ** 2 + nl ** 2), rtol=3e-5, atol=1e-8)
        assert np.isfinite(norm).all() and np.all(norm[~flag["hinge_active"]] == 0)
        assert np.all(norm[flag["hinge_active"]] > 0)
        close(float(summary["posterior_pair_gradient_mean_l2"]), norm.mean())
        close(float(summary["posterior_pair_gradient_max_l2"]), norm.max())
        vr = raw_by_split["val"][d]
        for dataset in ("KADID+TID", "KADID-10k", "TID2013"):
            subset = np.array([dataset == "KADID+TID" or r["dataset"] == dataset for r in val])
            raw_val = vr[subset]
            normalized_val = (raw_val - mean) / (std + 1e-8)
            r = next(r for r in val_stats if r["distance"] == d and r["dataset"] == dataset)
            assert int(r["n_images"]) == int(subset.sum())
            close(float(r["normalization_train_mean"]), mean)
            close(float(r["normalization_train_std_ddof0"]), std)
            check_stats(r, raw_val, "raw_")
            check_stats(r, normalized_val, "standardized_")
            _, counts = np.unique(normalized_val, return_counts=True)
            ties = int(np.sum(counts * (counts - 1) // 2))
            denominator = len(raw_val) * (len(raw_val) - 1) // 2
            assert int(r["exact_tied_unordered_pairs"]) == ties
            assert int(r["all_unordered_pairs"]) == denominator
            close(float(r["exact_tie_rate"]), ties / denominator)
            close(float(r["duplicate_score_image_fraction"]), np.sum(counts[counts > 1]) / len(raw_val))
    batches = rows(out / pole / "gradient_batch_contributions.csv")
    with np.load(out / pole / "full_mean_loss_gradients.npz", allow_pickle=False) as saved:
        for row in grad_stats:
            if row["status"].startswith("BLOCKED"):
                continue
            assert row["status"] == "PASS"
            d, p = row["distance"], row["parameter"]
            g = saved[d + "__" + p.replace(".", "_")]
            norm = np.linalg.norm(g)
            close(float(row["full_mean_loss_gradient_l2"]), norm)
            assert np.isfinite(g).all()
            if d == "D0" and p == "fc_log_var.weight":
                assert norm == 0 and row["zero_gradient_expected"] == "True"
            else:
                assert norm > 0 and row["zero_gradient_expected"] == "False"
            selected = [r for r in batches if r["distance"] == d and r["parameter"] == p]
            assert sum(int(r["n_images"]) for r in selected) == len(train)
            ns = column(selected, "contribution_gradient_l2")
            assert np.isfinite(ns).all()
            close(float(row["mean_gradient_norm"]), ns.mean())
            close(float(row["max_gradient_norm"]), ns.max())
            close(float(row["min_gradient_norm"]), ns.min())
            assert norm <= ns.sum() + 1e-10
    if pole == "EH":
        for table in (training_stats, gap_stats, val_stats, grad_stats):
            for d in ("KL", "W2_squared", "Bhattacharyya"):
                blocked = [r for r in table if r["distance"] == d]
                assert len(blocked) == 1 and blocked[0]["status"].startswith("BLOCKED")
                for key, value in blocked[0].items():
                    if key not in ("pole", "distance", "status", "reason"):
                        assert value == "", "A blocked metric must not contain a fabricated number"
        assert not (out / "EH/aggregate_reference.npz").exists()
    return {"status": "PASS", "verified_distances": list(names), "training_images": len(train),
            "validation_images": len(val), "pairs_per_distance": len(pairs),
            "independent_raw_formula_max_abs_errors": formula_errors,
            "independent_posterior_gradient_formulas": "PASS", "normalization": "PASS",
            "per_pair_gaps_ties_loss_and_summary": "PASS", "encoder_gradient_artifacts": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    out = args.output.resolve()
    source_hashes = json_file(out / "inputs/source_hashes.json")
    for p, value in source_hashes.items():
        assert sha(p) == value, f"Input changed: {p}"
    status = json_file(out / "preflight_status.json")
    assert status["status"] == "PARTIAL_EH_REFERENCE_BLOCKED" and not status["task_complete"]
    assert status["optimizer_steps"] == status["ranked_checkpoints_created"] == status["holdout_images_scored"] == 0
    assert not status["training_performed"] and not status["lambda_selected"]
    assert status["model_weights_unchanged"] and status["original_research_inputs_unchanged"]
    assert not status["ready_for_matched_distance_training"]
    train, val, pairs = (rows(out / "inputs" / n) for n in ("train_images.csv", "validation_images.csv", "pairs_used.csv"))
    assert (len(train), len(val), len(pairs)) == (9040, 1860, 18080)
    for data, split in ((train, "train"), (val, "val")):
        assert all(r["split"] == split and r["dataset"] in ("KADID-10k", "TID2013") for r in data)
        assert len({r["path"] for r in data}) == len(data)
        assert len({(r["dataset"], r["image_id"]) for r in data}) == len(data)
        for r in data:
            assert sha(r["path"]) == r["image_sha256"]
    assert not {r["path"] for r in train} & {r["path"] for r in val}
    assert not {(r["dataset"], r["ref_id"]) for r in train} & {(r["dataset"], r["ref_id"]) for r in val}
    index = {r["path"]: i for i, r in enumerate(train)}
    a, b = np.array([index[r["path_mild"]] for r in pairs]), np.array([index[r["path_severe"]] for r in pairs])
    result = {pole: verify_pole(out, pole, train, val, pairs, a, b) for pole in ("EA", "EH")}
    for pole in ("EA", "EH"):
        assert status[pole]["available_checks_passed"]
    eh = json_file(out / "EH_reference_manifest.json")
    assert not eh["original_reference_manifest_recovered"] and not eh["aggregate_reference_created"]
    assert eh["original_reference_count"] is None and all(not x["exists"] for x in eh["recovery_candidates"])
    import test_preflight
    stream = io.StringIO()
    run = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_preflight))
    (out / "verification/unit_tests.txt").write_text(stream.getvalue())
    assert run.wasSuccessful()
    result.update(status="EA_AND_EH_D0_VERIFIED_EH_AGGREGATE_BLOCKED", task_complete=False,
                  unit_tests_passed=run.testsRun, source_hashes_verified=len(source_hashes),
                  training_and_validation_image_hashes_verified=len(train) + len(val),
                  no_training_or_holdout_evaluation=True, verifier_sha256=sha(__file__))
    (out / "verification/verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "output_file_hashes.json")
    hashes = {str(p.relative_to(out)): sha(p) for p in files}
    (out / "verification/output_file_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
