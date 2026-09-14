"""Independent numeric, gradient, normalization, and sampling checks."""

import unittest

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from energies import EPS, d1_validity_probe, elbo_style, latent_energies, reference_diagonal, severity_selection
from run_ablation import DISTANCES, TID_NAMES, join_exact, normalize, pair_indices, row_correlation


class EnergyTests(unittest.TestCase):
    def test_known_distances(self):
        mu = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
        mr = torch.tensor([0.0, 1.0], dtype=torch.float64)
        vx = torch.tensor([[0.25, 4.0]], dtype=torch.float64)
        vr = torch.tensor([1.0, 0.25], dtype=torch.float64)
        out = latent_energies(mu, vx.log(), mr, vr)
        expected_d0 = np.sqrt(4 / (1 + EPS) + 4 / (0.25 + EPS))
        expected_d1 = np.sqrt(4 / (1.25 + EPS) + 4 / (1.25 + EPS))
        expected_d2 = 4 + 4 + (0.5 - 1) ** 2 + (2 - 0.5) ** 2
        expected_d3 = (4 / .625 + 4 / 2.125) / 8 + np.log((.625 * 2.125) / np.sqrt(.25 * 4 * 1 * .25)) / 2
        for key, expected in zip(("D0", "D1", "D2", "D3"), (expected_d0, expected_d1, expected_d2, expected_d3)):
            self.assertAlmostEqual(out[key].item(), expected, places=12)

    def test_identical_posteriors_and_variance_only_difference(self):
        mu = torch.tensor([[1., 2.]], dtype=torch.float64)
        variance = torch.tensor([.3, .8], dtype=torch.float64)
        equal = latent_energies(mu, variance.log().unsqueeze(0), mu[0], variance)
        for key in ("D0", "D1", "D2", "D3"):
            self.assertAlmostEqual(equal[key].item(), 0, places=12)
        probe = d1_validity_probe()
        self.assertTrue(probe["mu_only_false_changes_result"])
        self.assertTrue(probe["D1_responds_to_logvar_when_means_differ"])
        self.assertEqual(probe["D1_equal_means_unequal_variances"], 0)
        self.assertGreater(probe["D2_equal_means_unequal_variances"], 0)
        self.assertGreater(probe["D3_equal_means_unequal_variances"], 0)
        self.assertFalse(probe["D1_eligible_for_selection"])

    def test_gradients(self):
        mu = torch.tensor([[.2, -.3]], dtype=torch.float64, requires_grad=True)
        lv = torch.tensor([[-.8, -.9]], dtype=torch.float64, requires_grad=True)
        ref = torch.tensor([.7, .2], dtype=torch.float64)
        vr = torch.tensor([.6, .8], dtype=torch.float64)
        for key in ("D0", "D1", "D2", "D3"):
            self.assertTrue(torch.autograd.gradcheck(
                lambda m, l: latent_energies(m, l, ref, vr)[key], (mu, lv), atol=1e-5))
        stats = {"D0": {"mean": 1., "std": 2.}, "reconstruction_l1": {"mean": .2, "std": .05}}
        recon = torch.tensor([.1], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda m, r: elbo_style(latent_energies(m, lv, ref, vr)["D0"], r, stats), (mu, recon)))

    def test_tiny_variances_are_finite(self):
        scores = latent_energies(torch.tensor([[.1, -.2]]), torch.tensor([[-1000., -1000.]]),
                                 torch.zeros(2), torch.tensor([1e-12, 1e-12]))
        self.assertTrue(all(torch.isfinite(x).all() for x in scores.values()))

    def test_covariance_diagonal(self):
        a = torch.tensor([[1., .1], [.1, 2.]])
        torch.testing.assert_close(reference_diagonal(a, 2), torch.tensor([1., 2.]))
        with self.assertRaises(ValueError):
            reference_diagonal(torch.ones(3, 2), 2)

    def test_locked_selection_signs(self):
        r = {"blur": .8, "lens": .7, "sharpen": .3, "pixelate": -.2}
        self.assertAlmostEqual(severity_selection("EH", r), 1.6)
        self.assertAlmostEqual(severity_selection("EA", r), -1.8)
        self.assertEqual(TID_NAMES[7], "Gaussian blur")
        self.assertNotEqual(TID_NAMES[8], "Gaussian blur")

    def test_normalization_uses_supplied_stats(self):
        f = pd.DataFrame({"dataset": ["KADID-10k", "KADID-10k"]})
        stats = []
        for pole in ("EA", "EH"):
            for k in DISTANCES:
                f[f"{pole}_{k}"] = [100., 1000.]
                stats.append({"dataset": "KADID-10k", "pole": pole, "distance": k,
                              "mean": 10. if pole == "EA" else 20., "std_population": 2.})
        z = normalize(f, pd.DataFrame(stats))
        np.testing.assert_allclose(z.z_EA_D0, [45., 495.])
        np.testing.assert_allclose(z.Qz_D0, [5., 5.])

    def test_join_keys_include_dataset(self):
        a = pd.DataFrame({"dataset": ["KADID", "TID"], "image_id": ["I01", "I01"], "x": [1, 2]})
        b = pd.DataFrame({"dataset": ["TID", "KADID"], "image_id": ["I01", "I01"], "y": [20, 10]})
        result = join_exact(a, b).sort_values("x")
        self.assertEqual(result.y.tolist(), [10, 20])
        with self.assertRaises(ValueError):
            join_exact(a, b.iloc[:1])

    def test_gpu_float32_normalization_promotes_to_float64(self):
        f = pd.DataFrame({"dataset": ["KADID-10k", "KADID-10k"]})
        raw = np.array([3054.187744140625, 3127.9921875], dtype=np.float32)
        stats = []
        for pole in ("EA", "EH"):
            for distance in DISTANCES:
                f[f"{pole}_{distance}"] = raw
                stats.append({"dataset": "KADID-10k", "pole": pole, "distance": distance,
                              "mean": 3082.774338704427, "std_population": 28.546245644167072})
        z = normalize(f, pd.DataFrame(stats))
        expected = (raw.astype(np.float64)-3082.774338704427)/28.546245644167072
        np.testing.assert_allclose(z.z_EH_D0, expected, rtol=0, atol=1e-14)

    def test_pair_order_and_ties(self):
        f = pd.DataFrame({"dataset": ["KADID"] * 4, "ref_id": ["I01"] * 4,
                          "distortion_type": ["blur"] * 4, "severity_or_level": [2, 1, 3, 3]})
        mild, severe, _, _ = pair_indices(f)
        self.assertEqual(len(mild), 5)
        self.assertTrue((f.severity_or_level.to_numpy()[mild] < f.severity_or_level.to_numpy()[severe]).all())

    def test_bootstrap_ranks_recomputed_with_duplicate_clusters(self):
        from scipy.stats import rankdata
        x = np.array([[1, 3], [2, 2], [3, 4], [4, 1]], dtype=float)
        y = np.array([4, 2, 1, 3], dtype=float)
        idx = np.array([[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 2, 3]])
        px = row_correlation(x[idx], y[idx])
        sx = row_correlation(rankdata(x[idx], axis=1), rankdata(y[idx], axis=1))
        for i, ids in enumerate(idx):
            for j in range(2):
                self.assertAlmostEqual(px[i, j], pearsonr(x[ids, j], y[ids]).statistic)
                self.assertAlmostEqual(sx[i, j], spearmanr(x[ids, j], y[ids]).statistic)


if __name__ == "__main__":
    unittest.main(verbosity=2)
