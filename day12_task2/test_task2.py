"""Focused tests of the protocol's mathematics and split guards."""

import unittest
import numpy as np
import torch

from reference_math import EPS, DISTANCES, aggregate_posteriors, diagonal_distances
from run_task2 import EXPECTED_EA_COUNTS, validate_reference_rows, variance_only_probe


class ReferenceMathTests(unittest.TestCase):
    def test_population_total_variance(self):
        mu = np.array([[0., 1.], [2., 5.]])
        lv = np.log([[1., 4.], [3., 8.]])
        agg = aggregate_posteriors(mu, lv)
        np.testing.assert_allclose(agg["mu_agg"], [1., 3.])
        np.testing.assert_allclose(agg["v_between"], [1., 4.])
        np.testing.assert_allclose(agg["v_within"], [2., 6.])
        np.testing.assert_allclose(agg["v_agg"], [3., 10.])
        self.assertEqual(int(agg["ddof"]), 0)

    def test_lower_clamp_not_applied_to_between_component(self):
        agg = aggregate_posteriors(np.zeros((3, 2)), np.full((3, 2), -40.))
        np.testing.assert_array_equal(agg["v_between"], [0., 0.])
        np.testing.assert_array_equal(agg["v_agg"], [EPS, EPS])
        np.testing.assert_array_equal(agg["v_aggregate_raw"], agg["v_within"])

    def test_no_upper_variance_cap(self):
        agg = aggregate_posteriors(np.zeros((3, 2)), np.full((3, 2), np.log(7.)))
        np.testing.assert_allclose(agg["v_within"], [7., 7.])

    def test_reject_invalid_posteriors(self):
        for mu, lv in [(np.zeros((2, 2)), np.ones((2, 3))),
                       (np.zeros((1, 2)), np.zeros((1, 2))),
                       (np.full((2, 2), np.nan), np.zeros((2, 2)))]:
            with self.assertRaises(ValueError):
                aggregate_posteriors(mu, lv)

    def test_kl_matches_torch_distribution_and_direction(self):
        mu = torch.tensor([[1., -2.]], dtype=torch.float64)
        vx = torch.tensor([[.3, 4.]], dtype=torch.float64)
        rm = torch.tensor([.2, .1], dtype=torch.float64)
        vr = torch.tensor([3., .5], dtype=torch.float64)
        result = diagonal_distances(mu, vx.log(), rm, vr)["KL_q_to_reference"]
        q = torch.distributions.Normal(mu, vx.sqrt())
        r = torch.distributions.Normal(rm, vr.sqrt())
        expected = torch.distributions.kl_divergence(q, r).sum(-1)
        reverse = torch.distributions.kl_divergence(r, q).sum(-1)
        torch.testing.assert_close(result, expected)
        self.assertFalse(torch.allclose(result, reverse))

    def test_identity_and_variance_only_sensitivity(self):
        for values in variance_only_probe(100).values():
            self.assertAlmostEqual(values["matched_variance"], 0.)
            self.assertTrue(values["passed"])

    def test_w2_and_bhattacharyya_analytic(self):
        mu = torch.tensor([[0.]], dtype=torch.float64)
        r = diagonal_distances(mu, torch.tensor([[np.log(4.)]]), torch.zeros(1), torch.ones(1))
        self.assertAlmostEqual(r["W2_squared"].item(), 1., places=6)
        self.assertAlmostEqual(r["Bhattacharyya"].item(), .5 * np.log(1.25), places=6)

    def test_finite_difference_gradients(self):
        mu = torch.tensor([[.5, -.3]], dtype=torch.float64, requires_grad=True)
        lv = torch.tensor([[.3, -.8]], dtype=torch.float64, requires_grad=True)
        rm, rv = torch.zeros(2), torch.tensor([2., .5])
        for name in DISTANCES:
            self.assertTrue(torch.autograd.gradcheck(
                lambda m, l: diagonal_distances(m, l, rm, rv)[name], (mu, lv)))

    def test_reference_constants_are_detached(self):
        mu = torch.ones((2, 3), dtype=torch.float64, requires_grad=True)
        lv = torch.zeros_like(mu, requires_grad=True)
        rm = torch.zeros(3, requires_grad=True)
        rv = torch.full((3,), 2., requires_grad=True)
        sum(v.sum() for v in diagonal_distances(mu, lv, rm, rv).values()).backward()
        self.assertIsNone(rm.grad)
        self.assertIsNone(rv.grad)
        self.assertTrue(torch.isfinite(mu.grad).all())
        self.assertTrue(torch.isfinite(lv.grad).all())


class SplitGuardTests(unittest.TestCase):
    def setUp(self):
        self.refs, self.splits = [], []
        for dataset, n in EXPECTED_EA_COUNTS.items():
            for i in range(n):
                p = f"/tmp/task2_test_not_opened/{dataset}/{i}.png"
                self.refs.append({"path": p, "pool": dataset})
                self.splits.append({"distorted_path": p, "dataset": dataset,
                                    "image_id": str(i), "split": "train", "ref_path": "NA"})

    def test_exact_counts_and_cross_dataset_ids(self):
        rows = validate_reference_rows(self.refs, self.splits)
        self.assertEqual(len(rows), 2000)
        self.assertEqual(len({r["reference_id"] for r in rows}), 2000)

    def test_validation_or_holdout_rejected(self):
        for split in ("val", "holdout"):
            self.splits[0]["split"] = split
            with self.assertRaises(ValueError):
                validate_reference_rows(self.refs, self.splits)

    def test_duplicate_or_changed_pool_rejected(self):
        for rows in (self.refs + [self.refs[0]], self.refs[:-1]):
            with self.assertRaises(ValueError):
                validate_reference_rows(rows, self.splits)


if __name__ == "__main__":
    unittest.main(verbosity=2)
