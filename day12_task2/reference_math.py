"""Diagonal Gaussian distances for the clarified, matched-reference experiment."""

import numpy as np
import torch

EPS = 1e-8
DISTANCES = ("KL_q_to_reference", "W2_squared", "Bhattacharyya")


def aggregate_posteriors(mu, logvar):
    mu = np.asarray(mu, dtype=np.float64)
    logvar = np.asarray(logvar, dtype=np.float64)
    if mu.ndim != 2 or mu.shape != logvar.shape or len(mu) < 2:
        raise ValueError("Expected matching [N >= 2, latent_dim] posterior arrays")
    with np.errstate(over="raise", invalid="raise"):
        variance = np.exp(logvar)
    if not all(np.isfinite(a).all() for a in (mu, logvar, variance)):
        raise ValueError("Nonfinite posterior statistics")
    between = np.var(mu, axis=0, ddof=0)
    within = np.mean(variance, axis=0)
    raw = between + within
    return {
        "mu_agg": np.mean(mu, axis=0),
        "v_between": between,
        "v_within": within,
        "v_aggregate_raw": raw,
        "v_agg": np.maximum(raw, EPS),
        "variance_epsilon": np.array(EPS),
        "ddof": np.array(0),
        "n_images": np.array(len(mu)),
    }


def diagonal_distances(mu, logvar, reference_mu, reference_variance):
    """No upper variance cap; fixed lower clamp. Reference is a constant."""
    mu, logvar = mu.double(), logvar.double()
    ref_mu = reference_mu.detach().to(mu.device, dtype=torch.float64)
    ref_var = reference_variance.detach().to(mu.device, dtype=torch.float64)
    if mu.shape != logvar.shape or ref_mu.shape != mu.shape[-1:]:
        raise ValueError("Posterior/reference dimension mismatch")
    if ref_var.shape != ref_mu.shape:
        raise ValueError("Reference must be diagonal, one variance per dimension")
    vx = logvar.exp().clamp_min(EPS)
    vr = ref_var.clamp_min(EPS)
    delta2 = (mu - ref_mu).square()
    bar = (vx + vr) * 0.5
    return {
        "KL_q_to_reference": 0.5 * (vr.log() - vx.log() + (vx + delta2) / vr - 1).sum(-1),
        "W2_squared": (delta2 + (vx.sqrt() - vr.sqrt()).square()).sum(-1),
        "Bhattacharyya": (delta2 / bar).sum(-1) / 8
        + 0.5 * (bar.log() - 0.5 * (vx.log() + vr.log())).sum(-1),
    }
