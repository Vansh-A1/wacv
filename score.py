# Created by: Ankit Shukla
# Date : 21/07/2026
# S_Z latent-distance IQA score (WACV opinion-unaware CVAE, VGG+KLD).
#
# Orientation: HIGHER S_Z = WORSE quality (farther from pristine posterior).
# Also exposes REM (reconstruction error). DSM / discriminator-score is N/A.
#
# v1.2 hardening: mu_only (Mahalanobis on means) + Sigma_t clamp to stop
# degraded codes from shrinking S_Z via inflated posterior variance.
# Scoring defaults are loaded from configs/scoring_frozen.yaml (no silent drift).

import os
import torch

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


_SCORING_YAML_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "scoring_frozen.yaml"
)
_SCORING_CFG_CACHE = None


def load_scoring_config(path=None):
    """Load frozen scoring rule YAML. Returns dict (empty if missing / no PyYAML)."""
    global _SCORING_CFG_CACHE
    path = path or _SCORING_YAML_DEFAULT
    if _SCORING_CFG_CACHE is not None and path == _SCORING_YAML_DEFAULT:
        return _SCORING_CFG_CACHE
    cfg = {}
    if yaml is not None and os.path.isfile(path):
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    if path == _SCORING_YAML_DEFAULT:
        _SCORING_CFG_CACHE = cfg
    return cfg


def scoring_sz_kwargs(cfg=None, **overrides):
    """
    Kwargs for sz_from_stats / score_sz_* matching the frozen rule.
    Checkpoint-specific overrides may be passed explicitly.
    """
    cfg = cfg if cfg is not None else load_scoring_config()
    out = {
        "sigma_t_max": float(cfg.get("sz_sigma_t_max", 1.0)),
        "mu_only": (str(cfg.get("sz_mode", "mu_only")).lower() == "mu_only"),
    }
    out.update(overrides)
    return out


# ---------------------------------------------------------------------------
# S_Z core
# ---------------------------------------------------------------------------

def sz_from_stats(
    mu_t,
    log_var_t,
    mu_ref,
    Sigma_ref,
    eps=1e-8,
    sigma_t_max=None,
    mu_only=None,
):
    """
    S_Z = sqrt( sum_j (mu_ref_j - mu_t_j)^2 / (Sigma_ref_j + Sigma_t_j) )
    where Sigma_t = diag(exp(log_var)), optionally clamped / ignored.

    Hardening:
      - sigma_t_max: clamp posterior variance so uncertain degraded codes cannot
        artificially shrink S_Z via a huge Sigma_t (a common inversion cause).
      - mu_only: score with denominator = Sigma_ref only (Mahalanobis on means).

    Defaults for sigma_t_max / mu_only come from configs/scoring_frozen.yaml.
    """
    if sigma_t_max is None or mu_only is None:
        defaults = scoring_sz_kwargs()
        if sigma_t_max is None:
            sigma_t_max = defaults["sigma_t_max"]
        if mu_only is None:
            mu_only = defaults["mu_only"]

    squeeze = False
    if mu_t.dim() == 1:
        mu_t = mu_t.unsqueeze(0)
        log_var_t = log_var_t.unsqueeze(0)
        squeeze = True

    Sigma_t = torch.exp(log_var_t)
    if sigma_t_max is not None and sigma_t_max > 0:
        Sigma_t = torch.clamp(Sigma_t, max=float(sigma_t_max))

    mu_ref = mu_ref.to(mu_t.device).view(1, -1)
    Sigma_ref = Sigma_ref.to(mu_t.device).view(1, -1)

    num = (mu_ref - mu_t) ** 2
    if mu_only:
        den = Sigma_ref + eps
    else:
        den = Sigma_ref + Sigma_t + eps
    sz = torch.sqrt(torch.sum(num / den, dim=1))
    if squeeze:
        sz = sz.squeeze(0)
    return sz


def fit_reference(generator, dataloader, device, max_batches=None):
    """
    Encode pristine images; return mu_ref = mean(mu), Sigma_ref = diag(var(mu)).
    """
    generator.eval()
    mus = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device)
            _, mu, _ = generator(imgs)
            mus.append(mu.detach().cpu())
            if max_batches is not None and (i + 1) >= max_batches:
                break
    mus = torch.cat(mus, dim=0)  # (N, D)
    mu_ref = mus.mean(dim=0)
    Sigma_ref = mus.var(dim=0, unbiased=False).clamp_min(1e-8)
    return mu_ref, Sigma_ref


def ema_update_reference(mu_ref, Sigma_ref, mu_new, Sigma_new, momentum=0.99):
    """Exponential moving average of reference stats (on CPU tensors)."""
    mu_ref = momentum * mu_ref + (1.0 - momentum) * mu_new
    Sigma_ref = momentum * Sigma_ref + (1.0 - momentum) * Sigma_new
    return mu_ref, Sigma_ref.clamp_min(1e-8)


def save_reference(path, mu_ref, Sigma_ref):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "mu_ref": mu_ref.detach().cpu(),
            "Sigma_ref": Sigma_ref.detach().cpu(),
        },
        path,
    )


def load_reference(path, map_location="cpu"):
    d = torch.load(path, map_location=map_location, weights_only=False)
    return d["mu_ref"], d["Sigma_ref"]


@torch.no_grad()
def score_sz_eval(
    generator, x, mu_ref, Sigma_ref, sigma_t_max=None, mu_only=None
):
    """No-grad eval S_Z. x in [0,1], shape (B,3,H,W) or (3,H,W)."""
    was_training = generator.training
    generator.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    _, mu, log_var = generator(x)
    sz = sz_from_stats(
        mu, log_var, mu_ref, Sigma_ref, sigma_t_max=sigma_t_max, mu_only=mu_only
    )
    if was_training:
        generator.train()
    return sz


def score_sz_diff(
    generator, x, mu_ref, Sigma_ref, sigma_t_max=None, mu_only=None
):
    """
    Differentiable (posterior-mean) S_Z: uses encoder mu/log_var without
    reparameterization noise — gradients flow to x through the encoder.
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    _, mu, log_var = generator(x)
    return sz_from_stats(
        mu, log_var, mu_ref, Sigma_ref, sigma_t_max=sigma_t_max, mu_only=mu_only
    )


# ---------------------------------------------------------------------------
# Optional baseline: REM (reconstruction error). DSM omitted (no discriminator).
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_rem(generator, x, reduction="mean"):
    """Reconstruction-Error Metric (REM): pixel L1 between x and recon."""
    was_training = generator.training
    generator.eval()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    recon, _, _ = generator(x)
    err = torch.abs(recon - x)
    if reduction == "mean":
        rem = err.view(err.size(0), -1).mean(dim=1)
    else:
        rem = err.view(err.size(0), -1).sum(dim=1)
    if was_training:
        generator.train()
    return rem


def score_image(
    generator,
    x,
    mu_ref,
    Sigma_ref,
    metrics=("sz",),
    sigma_t_max=None,
    mu_only=None,
):
    """
    Score one batch/image. metrics subset of {'sz','rem'}.
    Returns dict of tensors (B,).
    """
    out = {}
    if "sz" in metrics:
        out["sz"] = score_sz_eval(
            generator, x, mu_ref, Sigma_ref, sigma_t_max=sigma_t_max, mu_only=mu_only
        )
    if "rem" in metrics:
        out["rem"] = score_rem(generator, x)
    return out
