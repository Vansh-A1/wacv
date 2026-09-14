#!/usr/bin/env python3
"""Independent NumPy audit of saved EA outputs and explicit EH blocked status."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from score import sz_from_stats


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def csv_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    out = HERE / "reference_stats"
    manifest = json.loads((out / "EA/manifest.json").read_text())
    for path, expected in manifest["source_hashes"].items():
        assert sha(path) == expected, f"Changed source {path}"
    for name, key in [("reference_posteriors.npz", "posterior_sha256"),
                      ("aggregate_reference.npz", "aggregate_sha256"),
                      ("legacy_reference.npz", "legacy_sha256"),
                      ("reference_images.csv", "image_manifest_sha256")]:
        assert sha(out / "EA" / name) == manifest[key], f"Changed output {name}"
    images = csv_rows(out / "EA/reference_images.csv")
    assert len(images) == 2000
    assert len({r["reference_id"] for r in images}) == 2000
    for row in images:
        assert sha(row["path"]) == row["image_sha256"], f"Changed image {row['path']}"
    with np.load(out / "EA/reference_posteriors.npz", allow_pickle=False) as p:
        mu, lv, vx = p["mu"], p["logvar"], p["variance"]
        assert list(p["reference_id"]) == [r["reference_id"] for r in images]
        assert list(p["path"]) == [r["path"] for r in images]
        assert str(p["checkpoint_sha256"]) == sha(manifest["checkpoint"])
        assert int(p["stored_epoch"]) == manifest["stored_epoch"] == 5
    assert mu.shape == lv.shape == vx.shape == (2000, 100)
    np.testing.assert_array_equal(vx, np.exp(lv))
    with np.load(out / "EA/aggregate_reference.npz", allow_pickle=False) as a:
        agg = {key: a[key] for key in a.files}
    independently_computed_mean = np.sum(mu, axis=0) / 2000
    between = np.sum((mu - independently_computed_mean) ** 2, axis=0) / 2000
    within = np.sum(np.exp(lv), axis=0) / 2000
    np.testing.assert_allclose(agg["mu_agg"], independently_computed_mean, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(agg["v_between"], between, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(agg["v_within"], within, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(agg["v_aggregate_raw"], between + within, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(agg["v_agg"], np.maximum(agg["v_aggregate_raw"], 1e-8))
    assert int(agg["ddof"]) == 0 and int(agg["n_images"]) == 2000
    assert all(np.isfinite(a).all() for a in (mu, lv, vx, *agg.values()))
    assert (agg["v_agg"] >= 1e-8).all()

    preserved = {}
    for pole in ("EA", "EH"):
        m = json.loads((out / pole / "manifest.json").read_text())
        ck = torch.load(m["checkpoint"], map_location="cpu", weights_only=False)
        with np.load(out / pole / "legacy_reference.npz", allow_pickle=False) as legacy:
            for key in ("mu_ref", "Sigma_ref"):
                expected = ck[key].numpy()
                assert expected.dtype == legacy[key].dtype and expected.shape == legacy[key].shape
                np.testing.assert_array_equal(expected, legacy[key])
        preserved[pole] = True

    dx2 = (mu - agg["mu_agg"]) ** 2
    vx, vr = np.maximum(vx, 1e-8), agg["v_agg"]
    average = (vx + vr) / 2
    expected_scores = {
        "KL_q_to_reference": .5 * np.sum(np.log(vr / vx) + (vx + dx2) / vr - 1, axis=1),
        "W2_squared": np.sum(dx2 + vx + vr - 2 * np.sqrt(vx * vr), axis=1),
        "Bhattacharyya": np.sum(dx2 / average, axis=1) / 8
        + .5 * np.sum(np.log(average / np.sqrt(vx * vr)), axis=1),
    }
    with np.load(out / "EA/legacy_reference.npz", allow_pickle=False) as old:
        expected_scores["D0_legacy"] = sz_from_stats(
            torch.from_numpy(mu).float(), torch.from_numpy(lv).float(),
            torch.from_numpy(old["mu_ref"]), torch.from_numpy(old["Sigma_ref"]),
            eps=1e-8, mu_only=True, sigma_t_max=1.).numpy()
    scores = csv_rows(out / "EA/reference_distance_scores.csv")
    assert [r["reference_id"] for r in scores] == [r["reference_id"] for r in images]
    for name, expected in expected_scores.items():
        actual = np.array([float(r[name]) for r in scores])
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
        assert np.isfinite(actual).all() and np.std(actual) > 0
    gradients = csv_rows(out / "gradient_preflight.csv")
    expected_checks = {(d, p) for d in expected_scores if d != "D0_legacy"
                       for p in ("enc1.weight", "fc_mu.weight", "fc_log_var.weight")}
    assert {(r["distance"], r["parameter"]) for r in gradients} == expected_checks
    assert len(gradients) == 9 and {r["pole"] for r in gradients} == {"EA"}
    assert all(r["passed"] == "True" and np.isfinite(float(r["gradient_l2"]))
               and float(r["gradient_l2"]) > 0 for r in gradients)
    status = json.loads((out / "task2_status.json").read_text())
    assert status["status"] == "PARTIAL"
    assert not (out / "EH/aggregate_reference.npz").exists()
    assert not (out / "EH/reference_posteriors.npz").exists()
    eh = json.loads((out / "EH/manifest.json").read_text())
    assert eh["reference_images_encoded"] == 0 and all(eh["ancestor_reference_tensor_equality"].values())

    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_task2.py")
    tests = unittest.TextTestRunner(verbosity=1).run(suite)
    assert tests.wasSuccessful()
    result = {
        "status": "EA_VERIFIED_EH_BLOCKED", "task_complete": False,
        "unit_tests_passed": tests.testsRun, "reference_images_verified": 2000,
        "independent_aggregate_recomputation": "PASS",
        "independent_distance_recomputation": "PASS",
        "checkpoint_and_image_hash_verification": "PASS",
        "original_legacy_tensor_equality": preserved,
        "encoder_gradient_checks_passed": len(gradients),
        "aggregate_variance_mean": float(agg["v_agg"].mean()),
        "between_variance_mean": float(between.mean()),
        "within_variance_mean": float(within.mean()),
        "verification_script_sha256": sha(__file__),
        "unit_test_script_sha256": sha(HERE / "test_task2.py"),
        "EH": "Exact original pristine-reference image list not available; no substitute used",
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
