"""Differentiable distance candidates for the frozen Day-12 Task-1 ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from score import sz_from_stats

EPS = 1e-8
SIGMA_T_MAX = 1.0
DISTANCES = ("D0", "D1", "D2", "D3", "D4")
ELIGIBLE = ("D0", "D2", "D3", "D4")


def reference_diagonal(covariance: torch.Tensor, dimensions: int) -> torch.Tensor:
    if covariance.shape == (dimensions,):
        return covariance
    if covariance.shape == (dimensions, dimensions):
        return torch.diagonal(covariance)
    raise ValueError(f"Unexpected reference covariance shape: {covariance.shape}")


def latent_energies(mu, logvar, mu_ref, covariance):
    """D0/D1 call score.py exactly; D2 is squared W2, as specified."""
    if covariance.ndim != 1:
        raise ValueError("Existing D0/D1 flatten Sigma_ref: a full matrix is unsupported.")
    d0 = sz_from_stats(mu, logvar, mu_ref, covariance, eps=EPS,
                       sigma_t_max=SIGMA_T_MAX, mu_only=True)
    d1 = sz_from_stats(mu, logvar, mu_ref, covariance, eps=EPS,
                       sigma_t_max=SIGMA_T_MAX, mu_only=False)
    # Double precision avoids cancellation in the Bhattacharyya variance term.
    delta = mu.double() - mu_ref.double()
    vx = logvar.double().exp().clamp_min(EPS)
    vr = reference_diagonal(covariance, mu.shape[-1]).double().clamp_min(EPS)
    bar = (vx + vr) / 2
    d2 = (delta.square() + (vx.sqrt() - vr.sqrt()).square()).sum(dim=-1)
    d3 = (delta.square() / bar).sum(dim=-1) / 8
    d3 = d3 + (bar.log() - (vx.log() + vr.log()) / 2).sum(dim=-1) / 2
    return {"D0": d0, "D1": d1, "D2": d2, "D3": d3}


def elbo_style(d0, reconstruction_l1, train_stats):
    z0 = (d0 - train_stats["D0"]["mean"]) / (train_stats["D0"]["std"] + EPS)
    zr = (reconstruction_l1 - train_stats["reconstruction_l1"]["mean"]) / (
        train_stats["reconstruction_l1"]["std"] + EPS)
    return z0 + zr


def severity_selection(pole, correlations):
    if pole == "EH":
        return sum(correlations[k] for k in ("blur", "lens", "sharpen", "pixelate"))
    if pole == "EA":
        return (-correlations["blur"] - correlations["lens"]
                - max(0.0, correlations["sharpen"])
                - max(0.0, correlations["pixelate"]))
    raise ValueError(pole)


def d1_validity_probe():
    mu = torch.zeros((1, 2), dtype=torch.float64)
    ref = torch.zeros(2, dtype=torch.float64)
    vr = torch.tensor([0.1, 0.2], dtype=torch.float64)
    lv_equal = vr.log().unsqueeze(0)
    lv_different = (vr * 2).log().unsqueeze(0)
    equal = latent_energies(mu, lv_equal, ref, vr)
    different = latent_energies(mu, lv_different, ref, vr)
    moved = torch.ones_like(mu)
    a = latent_energies(moved, lv_equal, ref, vr)
    b = latent_energies(moved, lv_different, ref, vr)
    return {
        "mu_only_false_changes_result": bool(a["D0"].item() != a["D1"].item()),
        "D1_responds_to_logvar_when_means_differ": bool(a["D1"].item() != b["D1"].item()),
        "D1_identical_distributions": equal["D1"].item(),
        "D1_equal_means_unequal_variances": different["D1"].item(),
        "D2_equal_means_unequal_variances": different["D2"].item(),
        "D3_equal_means_unequal_variances": different["D3"].item(),
        "D1_eligible_for_selection": False,
        "reason": "Variance changes only the mean-distance denominator. Equal means give zero "
                  "even with unequal variances, so D1 fails the requested full-posterior "
                  "difference interpretation. Computed and retained as a diagnostic only.",
    }
