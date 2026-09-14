"""Task 3 tests plus the verified Task 2 formula/preprocessing suite."""

import ast
from pathlib import Path
import unittest
import numpy as np
import torch
from preflight_core import (energies, gap, normalize, summarize, pair_arrays,
                            validation_summary, loss_coefficients, locked_membership,
                            MARGIN, NORM_EPS)
import test_task2


class PreflightTests(unittest.TestCase):
    def test_population_stats_unique_images(self):
        x = np.array([1., 3., 8.])
        s = summarize(x)
        self.assertEqual(s["std_ddof0"], np.std(x, ddof=0))
        self.assertNotEqual(s["std_ddof0"], np.std(x, ddof=1))
        np.testing.assert_allclose(normalize(x, s["mean"], s["std_ddof0"]),
                                   (x - x.mean()) / (x.std() + 1e-8))

    def test_validation_uses_training_stats(self):
        r = validation_summary(np.array([10., 12., 14.]), mean=2., std=2.)
        self.assertAlmostEqual(r["standardized_mean"], 10 / (2 + 1e-8))
        self.assertNotAlmostEqual(r["standardized_mean"], 0)

    def test_opposite_polarities(self):
        x = np.array([5., 1., 3.])
        a, b = np.array([0, 1, 0]), np.array([1, 2, 2])
        ea, eh = (pair_arrays(x, 0., 1., a, b, p) for p in ("EA", "EH"))
        np.testing.assert_array_equal(ea["gap"], -eh["gap"])
        np.testing.assert_array_equal(ea["strict_correct"], [True, False, True])

    def test_ties_separate_from_negligible(self):
        x = np.array([0., 0., 5e-7, 2e-6])
        r = pair_arrays(x, 0, 1, np.array([0, 0, 0]), np.array([1, 2, 3]), "EH")
        np.testing.assert_array_equal(r["exact_tie"], [True, False, False])
        np.testing.assert_array_equal(r["negligible_gap"], [True, True, False])
        np.testing.assert_array_equal(r["strict_correct"], [False, True, True])

    def test_margin_and_inactive_pairs(self):
        x = np.array([0., .2, -.2]) * (1 + NORM_EPS)
        r = pair_arrays(x, 0, 1, np.array([0, 0]), np.array([1, 2]), "EH")
        np.testing.assert_allclose(r["rank_loss"], [0, .3])
        np.testing.assert_array_equal(r["hinge_active"], [False, True])

    def test_validation_exact_tie_denominator(self):
        r = validation_summary(np.array([1., 1., 2., 2.]), 0, 1)
        self.assertEqual(r["exact_tied_unordered_pairs"], 2)
        self.assertEqual(r["all_unordered_pairs"], 6)
        self.assertAlmostEqual(r["exact_tie_rate"], 1 / 3)
        self.assertEqual(r["duplicate_score_image_fraction"], 1.)

    def test_d0_ignores_variance_without_false_failure(self):
        m = torch.tensor([[1., 2.], [2., 4.]], requires_grad=True)
        l = torch.zeros_like(m, requires_grad=True)
        legacy = {"mu_ref": torch.zeros(2), "Sigma_ref": torch.ones(2)}
        dm, dl = torch.autograd.grad(energies(m, l, legacy)["D0"].sum(), (m, l), allow_unused=True)
        self.assertIsNone(dl)
        self.assertTrue(torch.isfinite(dm).all())
        self.assertTrue(torch.any(dm != 0))

    def test_chain_accumulation_matches_direct_full_pair_loss(self):
        torch.manual_seed(7)
        x = torch.randn(7, 3, dtype=torch.float64)
        weight = torch.randn(3, dtype=torch.float64, requires_grad=True)
        a, b = np.array([0, 0, 1, 2, 2, 3]), np.array([1, 2, 2, 3, 4, 5])
        for pole in ("EA", "EH"):
            raw = (x @ weight).square()
            mean, std = raw.detach().mean().item(), raw.detach().std(unbiased=False).item()
            z = normalize(raw, mean, std)
            loss = torch.relu(MARGIN - gap(z[a], z[b], pole)).mean()
            direct = torch.autograd.grad(loss, weight)[0]
            coeff, value = loss_coefficients(raw.detach().numpy(), mean, std, a, b, pole)
            self.assertAlmostEqual(value, loss.item())
            accumulated = torch.zeros_like(weight)
            for start in range(0, len(x), 2):
                score = (x[start:start+2] @ weight).square()
                accumulated += torch.autograd.grad(score, weight, grad_outputs=coeff[start:start+2])[0]
            torch.testing.assert_close(accumulated, direct, rtol=1e-12, atol=1e-12)

    def test_no_optimizer_or_checkpoint_saving(self):
        for name in ("run_preflight.py", "preflight_core.py"):
            tree = ast.parse((Path(__file__).parent / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, ("optim", "step", "backward"))
                    self.assertFalse(node.attr == "save" and isinstance(node.value, ast.Name)
                                     and node.value.id == "torch")

    def fixture(self):
        rows = []
        for split, ref, levels in (("train", "R1", (1, 2)), ("val", "R2", (1,)), ("holdout", "R3", (1,))):
            for sev in levels:
                rows.append({"dataset": "KADID-10k", "image_id": f"{ref}_{sev}",
                             "ref_id": ref, "distortion_type": "Gaussian blur", "split": split,
                             "severity_or_level": str(sev), "ref_path": f"/tmp/{ref}.png",
                             "distorted_path": f"/tmp/{ref}_{sev}.png"})
        pair = {"dataset": "KADID-10k", "ref_id": "R1", "distortion_type": "Gaussian blur",
                "image_id_mild": "R1_1", "image_id_severe": "R1_2", "sev_mild": "1", "sev_severe": "2",
                "path_mild": "/tmp/R1_1.png", "path_severe": "/tmp/R1_2.png"}
        return rows, [pair]

    def test_valid_locked_membership(self):
        train, val, pairs, a, b = locked_membership(*self.fixture())
        self.assertEqual((len(train), len(val), len(pairs)), (2, 1, 1))
        np.testing.assert_array_equal(a, [0])
        np.testing.assert_array_equal(b, [1])

    def test_validation_leak_is_rejected(self):
        rows, pairs = self.fixture()
        pairs[0]["path_mild"] = "/tmp/R2_1.png"
        with self.assertRaises(ValueError):
            locked_membership(rows, pairs)

    def test_duplicate_pairs_rejected(self):
        rows, pairs = self.fixture()
        with self.assertRaises(ValueError):
            locked_membership(rows, pairs * 2)

    def test_wrong_severity_rejected(self):
        rows, pairs = self.fixture()
        pairs[0]["sev_severe"] = "1"
        with self.assertRaises(ValueError):
            locked_membership(rows, pairs)

    def test_nonfinite_stats_are_visible(self):
        r = summarize([1., np.inf, np.nan])
        self.assertEqual(r["nonfinite_count"], 2)
        self.assertEqual(r["finite_count"], 1)
        self.assertFalse(r["nonconstant"])


def load_tests(loader, tests, pattern):
    tests.addTests(loader.loadTestsFromModule(test_task2))
    return tests


if __name__ == "__main__":
    unittest.main()
