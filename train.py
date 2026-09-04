# Created by: Ankit Shukla
# Date : 21/07/2026
# Train CVAE ONLY (VGG+KLD, no discriminator) — WACV opinion-unaware IQA.
# Fixes from old code: MEAN KL + warmup, VGG ImageNet-normalized features,
# ldim actually used, S_Z separation as default best metric, deterministic
# seeding + resume. No adversarial / discriminator losses.
# v1.2+: frozen mu_only S_Z, ranking loss, larger eval set (stops S_Z collapse).
# v1.4+: matched-ladder eval, multi-crop S_Z aggregation, richer ranking degradations.
# v1.5+: versioned TRAIN/VAL/TEST pools; kadid_srcc checkpoint selection; expanded
# ranking degradations (sharpen/color/pixelate); Kendall ladder diagnostic.

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnnP
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import vgg19

warnings.filterwarnings("ignore", category=UserWarning)

_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "external"))
from dataloader import (  # noqa: E402
    ManifestImageDataset,
    EvalImageDataset,
    center_crop,
    _resize_short_side,
)
from model import CVAEGenerator_v2  # noqa: E402

from score import (  # noqa: E402
    ema_update_reference,
    fit_reference,
    load_reference,
    save_reference,
    sz_from_stats,
)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Student package: pools are provided via --train_list (absolute paths).
HOLD_OUT_POOLS = set()
DEFAULT_TRAIN_POOLS = ""
DEFAULT_EVAL_ROOTS = ""


# ---------------------------------------------------------------------------
# helpers (style-matched)
# ---------------------------------------------------------------------------

def save_intermediateResults(image1, image2, imageSavePath):
    width1, height1 = image1.size
    total_width = width1 + image2.size[0]
    merged_image = Image.new("RGB", (total_width, height1))
    merged_image.paste(image1, (0, 0))
    merged_image.paste(image2, (width1, 0))
    merged_image.save(imageSavePath)


def get_git_hash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnnP.deterministic = True
    cudnnP.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def pools_tag(pools):
    abbrev = {
        "KADIS-700k": "kadis",
        "KADID-10k": "kadid",
        "CelebA-HQ-train": "celeba",
        "CelebA-HQ-val": "celebaval",
        "LIVE": "live",
        "DIV2K-train": "div2k",
        "DIV2K-valid": "div2kval",
    }
    return "-".join(abbrev.get(p, p.replace(" ", "")) for p in pools)


def read_manifest(path):
    paths = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not os.path.isabs(line):
                line = os.path.join(_ROOT, line)
            paths.append(line)
    return paths


def version_tag(ver):
    """Normalize --ver into a stable run-name token (no timestamps)."""
    v = str(ver).strip()
    if not v:
        v = "1.0"
    if not v.lower().startswith("v"):
        v = f"v{v}"
    return v


def make_run_name(opt, train_pools):
    return f"cvae_v2_ld{opt.ldim}_{opt.img}_{pools_tag(train_pools)}_{version_tag(opt.ver)}"


def run_already_started(ckpt_dir, run_dir):
    if os.path.isfile(os.path.join(ckpt_dir, "last.pth")):
        return True
    if os.path.isfile(os.path.join(run_dir, "metrics.csv")):
        return True
    return False


# ---------------------------------------------------------------------------
# VGG perceptual loss (relu1_2, relu2_2, relu3_3, relu4_3) + ImageNet norm
# ---------------------------------------------------------------------------

class VGGPerceptualLoss(nn.Module):
    # VGG19 feature indices (0-based into .features):
    # relu1_2=3, relu2_2=8, relu3_3=17, relu4_3=26
    LAYER_IDS = (3, 8, 17, 26)

    def __init__(self, device):
        super().__init__()
        try:
            vgg = vgg19(weights="DEFAULT").features.to(device).eval()
        except TypeError:
            vgg = vgg19(pretrained=True).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.slices = nn.ModuleList()
        prev = 0
        for idx in self.LAYER_IDS:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, idx + 1)]))
            prev = idx + 1
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, recon, target):
        x = self.normalize(recon)
        y = self.normalize(target)
        loss = 0.0
        for sl in self.slices:
            x = sl(x)
            y = sl(y)
            loss = loss + F.mse_loss(x, y, reduction="mean")
        return loss / float(len(self.slices))


def kl_loss(mu, log_var):
    # L_KL = -0.5 * mean( sum_j (1 + log_var - mu^2 - exp(log_var)) )
    return -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))


def kl_warmup_beta(step, warmup_steps, beta_kl):
    if warmup_steps <= 0:
        return beta_kl
    return beta_kl * min(1.0, float(step) / float(warmup_steps))


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def psnr_batch(pred, target, eps=1e-8):
    mse = F.mse_loss(pred, target, reduction="none").view(pred.size(0), -1).mean(dim=1)
    return (10.0 * torch.log10(1.0 / (mse + eps))).mean().item()


def ssim_batch(pred, target):
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure

        return structural_similarity_index_measure(pred, target, data_range=1.0).item()
    except Exception:
        pass
    try:
        import piq

        return piq.ssim(pred, target, data_range=1.0).item()
    except Exception:
        return (1.0 - F.l1_loss(pred, target)).item()


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="Train CVAE (VGG+KLD) for opinion-unaware IQA")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--ver",
        type=str,
        default="ft1",
        help="version tag in run_name (e.g. ft1 -> ..._vft1). Bump for a new run; use --resume to continue.",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    # v1.1+: lower LR — high LR + long recon training collapses S_Z orientation
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.5, 0.999])
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step"],
    )
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="stop after N validations with no best-metric improvement (0 disables)",
    )
    parser.add_argument(
        "--init_from",
        type=str,
        default="checkpoints/best.pth",
        help="warm-start generator weights from a prior best.pth (fresh optimizer)",
    )
    parser.add_argument(
        "--init_ref_from",
        type=str,
        default="",
        help="optional reference_stats.pt to use with warm-started weights; overrides mu_ref/Sigma_ref embedded in --init_from",
    )

    parser.add_argument("--ldim", default=100, type=int, help="latent dimension")
    parser.add_argument("--img", default=256, type=int, help="crop / image size")
    parser.add_argument("--patch", default=256, type=int, help="alias of --img if set")

    parser.add_argument(
        "--pools",
        type=str,
        default=DEFAULT_TRAIN_POOLS,
        help="comma list of training pools (ignored if --train_list is set)",
    )
    parser.add_argument("--manifest_dir", type=str, default="manifests")
    parser.add_argument(
        "--train_list",
        type=str,
        default=None,
        help="text file with one absolute pristine image path per line (required)",
    )
    parser.add_argument("--eval_pool", type=str, default="student_eval")
    parser.add_argument(
        "--eval_root",
        type=str,
        default=None,
        help="legacy single eval root (overrides --eval_roots if set)",
    )
    parser.add_argument(
        "--eval_roots",
        type=str,
        default=DEFAULT_EVAL_ROOTS,
        help="comma-separated eval roots (HR+LR+gblur). Larger set => stabler AUROC",
    )
    parser.add_argument(
        "--eval_list",
        type=str,
        default="examples/eval_list_example.txt",
        help="explicit eval file list (overrides --eval_roots walk)",
    )
    parser.add_argument(
        "--ladder_meta",
        type=str,
        default="",
        help="optional ladder_meta.json (leave empty if you do not have ladders)",
    )
    parser.add_argument(
        "--kadid_monitor_csv",
        type=str,
        default="",
        help="optional KADID MOS CSV for kadid_srcc best_metric",
    )
    parser.add_argument(
        "--kadid_monitor_max",
        type=int,
        default=None,
        help="optional cap on monitor images scored each validation (None=all)",
    )
    parser.add_argument(
        "--n_eval_crops",
        type=int,
        default=5,
        help="aggregate S_Z over N crops per eval image (1=center only; v1.5 default 5)",
    )

    parser.add_argument("--pix_loss", type=str, default="l1", choices=["l1", "mse"])
    parser.add_argument(
        "--w_pix",
        type=float,
        default=1.0,
        help="weight on pixel recon; lower (e.g. 0.5) reduces over-recon that collapses S_Z",
    )
    parser.add_argument("--beta_kl", type=float, default=1.0)
    parser.add_argument("--kl_warmup_steps", type=int, default=5000)
    parser.add_argument(
        "--free_bits",
        type=float,
        default=1.0,
        help="clamp KL from below (free-bits). 0 disables. Helps keep S_Z oriented correctly.",
    )
    parser.add_argument("--w_vgg", type=float, default=0.1)
    parser.add_argument(
        "--ref_mode",
        type=str,
        default="once",
        choices=["once", "every", "ema"],
        help="S_Z reference refresh: once=freeze after first fit (recommended)",
    )
    parser.add_argument("--ref_ema", type=float, default=0.99, help="EMA momentum if ref_mode=ema")
    parser.add_argument(
        "--sz_mode",
        type=str,
        default="mu_only",
        choices=["full", "mu_only"],
        help="S_Z denominator: full=Sigma_ref+Sigma_t, mu_only=Sigma_ref only",
    )
    parser.add_argument(
        "--sz_sigma_t_max",
        type=float,
        default=1.0,
        help="clamp encoder Sigma_t when sz_mode=full",
    )
    parser.add_argument(
        "--lambda_rank",
        type=float,
        default=0.25,
        help="weight for clean-vs-degraded S_Z ranking loss (0 disables)",
    )
    parser.add_argument(
        "--rank_margin",
        type=float,
        default=0.5,
        help="margin for ranking: softplus(margin + sz_clean - sz_deg)",
    )
    parser.add_argument(
        "--auto_balance",
        action="store_true",
        default=False,
        help="one-shot grad-norm reweight of w_vgg (off by default)",
    )
    parser.add_argument("--no_auto_balance", action="store_true", default=False)
    parser.add_argument("--balance_w_vgg_min", type=float, default=0.05)
    parser.add_argument("--balance_w_vgg_max", type=float, default=1.0)

    parser.add_argument("--val_every", type=int, default=10, help="validate every N epochs")
    parser.add_argument(
        "--save_epoch_every",
        type=int,
        default=0,
        help="save epoch_XXXX.pth every N epochs in addition to last.pth; 0 disables",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="separation",
        choices=["separation", "psnr", "lpips", "ladder_spearman", "kadid_srcc"],
        help="metric used to pick best.pth (separation works without MOS labels)",
    )
    parser.add_argument(
        "--latent_plot",
        type=str,
        default="both",
        choices=["pca", "tsne", "both"],
    )
    parser.add_argument("--tsne_max", type=int, default=500)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)

    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--run_name", type=str, default=None, help="override auto run_name")
    parser.add_argument(
        "--output_root",
        type=str,
        default=".",
        help="root directory for checkpoints, runs, and configs",
    )

    parser.add_argument("--ref_max_batches", type=int, default=None)
    parser.add_argument(
        "--max_train_images",
        type=int,
        default=None,
        help="optional cap (keeps non-KADIS first, then fills from KADIS)",
    )
    parser.add_argument("--grad_log_every", type=int, default=100)
    parser.add_argument(
        "--train_recon_every",
        type=int,
        default=200,
        help="save a train input|recon PNG every N iters",
    )

    opt = parser.parse_args()
    if opt.no_auto_balance:
        opt.auto_balance = False
    if opt.patch != 256 and opt.img == 256:
        opt.img = opt.patch
    # legacy: --eval_root alone overrides multi-root default
    if opt.eval_root:
        opt.eval_roots = opt.eval_root
    return opt


def sz_kwargs(opt):
    return {
        "sigma_t_max": opt.sz_sigma_t_max,
        "mu_only": (opt.sz_mode == "mu_only"),
    }


def synthesize_degraded(x):
    """Synthetic degradation for ranking loss (covers KADID failure-mode families)."""
    r = random.random()
    # blur / smooth (~30%)
    if r < 0.18:
        k = random.choice([5, 9, 15, 21])
        pad = k // 2
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
    if r < 0.30:
        y = x
        for _ in range(random.choice([2, 3, 4])):
            y = F.avg_pool2d(y, kernel_size=5, stride=1, padding=2)
        return y
    # downsample-upsample (~18%)
    if r < 0.48:
        scale = random.choice([4, 8, 16])
        h, w = x.shape[-2:]
        small = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    # noise + quantization (~14%)
    if r < 0.62:
        noise = torch.randn_like(x) * random.uniform(0.02, 0.08)
        y = (x + noise).clamp(0, 1)
        levels = random.choice([8, 16, 32])
        return (torch.round(y * levels) / float(levels)).clamp(0, 1)
    # oversharpen via high-pass boost (~12%)
    if r < 0.74:
        blur = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        amount = random.uniform(1.5, 5.0)
        return (x + amount * (x - blur)).clamp(0, 1)
    # color / tone jitter (~14%)
    if r < 0.88:
        # per-channel gain + bias away from identity
        gain = 0.4 + 0.6 * random.random()  # [0.4, 1.0]
        if random.random() < 0.5:
            gain = 1.0 / max(gain, 1e-3)  # brighten path too
            gain = min(gain, 1.8)
        bias = random.uniform(-0.15, 0.15)
        return (x * gain + bias).clamp(0, 1)
    # nearest-neighbor pixelation (~12%)
    block = random.choice([4, 8, 16, 32])
    h, w = x.shape[-2:]
    sh, sw = max(1, h // block), max(1, w // block)
    small = F.interpolate(x, size=(sh, sw), mode="nearest")
    return F.interpolate(small, size=(h, w), mode="nearest")


def ranking_sz_loss(sz_clean, sz_deg, margin):
    """Encourage sz_deg >= sz_clean + margin (higher S_Z = worse)."""
    return F.softplus(margin + sz_clean - sz_deg).mean()


def collect_eval_files(eval_roots_csv):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = []
    roots = [r.strip() for r in eval_roots_csv.split(",") if r.strip()]
    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] eval root missing: {root}")
            continue
        for dirpath, _, fns in os.walk(root):
            for fn in sorted(fns):
                if os.path.splitext(fn)[1].lower() in exts:
                    files.append(os.path.join(dirpath, fn))
    return sorted(files)


def load_eval_list(path):
    if not path or not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            # allow relative paths from the package root
            if not os.path.isabs(ln):
                ln = os.path.join(_ROOT, ln)
            if os.path.isfile(ln):
                out.append(ln)
    return out


def score_image_multicrop(
    generator, path, crop_size, transform, device, mu_ref, Sigma_ref, skw, n_crops, seed
):
    """Arithmetic mean of S_Z over 1 center crop + (n_crops-1) seeded random crops."""
    pil = Image.open(path).convert("RGB")
    pil = _resize_short_side(pil, crop_size)
    crops = [center_crop(pil, crop_size)]
    if n_crops > 1:
        rng = random.Random(seed)
        w, h = pil.size
        for _ in range(n_crops - 1):
            max_x = max(0, w - crop_size)
            max_y = max(0, h - crop_size)
            x0 = rng.randint(0, max_x) if max_x > 0 else 0
            y0 = rng.randint(0, max_y) if max_y > 0 else 0
            crops.append(pil.crop((x0, y0, x0 + crop_size, y0 + crop_size)))
    szs = []
    mus = []
    recon0 = None
    x0 = None
    for i, c in enumerate(crops):
        arr = (np.asarray(c) / 255.0).astype("float32")
        x = transform(arr).unsqueeze(0).to(device)
        recon, mu, log_var = generator(x)
        sz = sz_from_stats(mu, log_var, mu_ref, Sigma_ref, **skw)
        szs.append(float(sz.view(-1)[0].item()))
        mus.append(mu.detach().cpu())
        if i == 0:
            recon0, x0 = recon, x
    return float(np.mean(szs)), torch.cat(mus, dim=0).mean(dim=0, keepdim=True), x0, recon0


def eval_ladder_spearman(generator, ladder_meta_path, transform, device, mu_ref, Sigma_ref, opt):
    """Per-image Kendall tau of S_Z vs severity (more stable than Spearman of 4 level-means)."""
    empty = {
        "blur_rho": float("nan"),
        "jpeg_rho": float("nan"),
        "lr_rho": float("nan"),
        "sharpen_rho": float("nan"),
        "color_rho": float("nan"),
        "pixelate_rho": float("nan"),
        "ladder_spearman": float("nan"),
    }
    if not ladder_meta_path or not os.path.isfile(ladder_meta_path):
        return empty
    with open(ladder_meta_path) as f:
        meta = json.load(f)
    skw = sz_kwargs(opt)

    def _rho(mapping, higher_severity_worse=True):
        # mapping: severity_str -> [paths]
        # Score every image individually, then Kendall-tau vs severity rank.
        pairs = []
        levels = sorted(mapping.keys(), key=lambda k: float(k))
        if not higher_severity_worse:
            levels = sorted(mapping.keys(), key=lambda k: -float(k))
        for i, lev in enumerate(levels):
            for p in mapping[lev]:
                if not os.path.isfile(p):
                    continue
                sz, _, _, _ = score_image_multicrop(
                    generator,
                    p,
                    opt.img,
                    transform,
                    device,
                    mu_ref,
                    Sigma_ref,
                    skw,
                    n_crops=1,  # center only — ladder has many images; keep val fast
                    seed=opt.seed + hash(os.path.basename(p)) % 10007,
                )
                pairs.append((float(i), float(sz)))
        if len(pairs) < 6:
            return float("nan")
        xs = [a for a, _ in pairs]
        ys = [b for _, b in pairs]
        # Kendall is more stable on noisy ordinal data than Spearman-of-means
        tau, _ = kendalltau(xs, ys)
        return float(tau) if tau is not None else float("nan")

    blur_rho = _rho(meta.get("blur", {}), higher_severity_worse=True)
    jpeg_rho = _rho(meta.get("jpeg", {}), higher_severity_worse=False)
    lr_rho = _rho(meta.get("lr", {}), higher_severity_worse=True)
    sharpen_rho = _rho(meta.get("sharpen", {}), higher_severity_worse=True)
    color_rho = _rho(meta.get("color", {}), higher_severity_worse=False)  # lower factor = worse
    pixelate_rho = _rho(meta.get("pixelate", {}), higher_severity_worse=True)
    vals = [
        v
        for v in (blur_rho, jpeg_rho, lr_rho, sharpen_rho, color_rho, pixelate_rho)
        if np.isfinite(v)
    ]
    ladder = float(np.mean(vals)) if vals else float("nan")
    return {
        "blur_rho": blur_rho,
        "jpeg_rho": jpeg_rho,
        "lr_rho": lr_rho,
        "sharpen_rho": sharpen_rho,
        "color_rho": color_rho,
        "pixelate_rho": pixelate_rho,
        "ladder_spearman": ladder,  # name kept for TensorBoard compat; value is mean Kendall
    }


@torch.no_grad()
def eval_kadid_monitor(generator, csv_path, transform, device, mu_ref, Sigma_ref, opt):
    """
    Mean per-distortion-type SRCC(S_Z, MOS) on the KADID monitor split.
    S_Z higher=worse, MOS higher=better => expect negative SRCC; we return the signed
    mean SRCC (more negative = better) as kadid_srcc for checkpoint selection.
    """
    out = {
        "kadid_srcc": float("nan"),
        "kadid_n": 0,
        "kadid_n_types": 0,
    }
    if not csv_path or not os.path.isfile(csv_path):
        return out

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append((r["path"], r["ref_img"], float(r["mos"])))
    if opt.kadid_monitor_max is not None and len(rows) > opt.kadid_monitor_max:
        rng = np.random.RandomState(opt.seed)
        idx = rng.choice(len(rows), size=opt.kadid_monitor_max, replace=False)
        rows = [rows[i] for i in sorted(idx.tolist())]

    skw = sz_kwargs(opt)
    by_dist = {}
    for i, (path, ref, mos) in enumerate(rows):
        if not os.path.isfile(path):
            continue
        bn = os.path.basename(path)
        # Ixx_yy_zz.png -> yy is distortion type
        parts = bn.replace(".png", "").split("_")
        if len(parts) < 3:
            continue
        dist_id = parts[1]
        sz, _, _, _ = score_image_multicrop(
            generator,
            path,
            opt.img,
            transform,
            device,
            mu_ref,
            Sigma_ref,
            skw,
            n_crops=1,  # center only for speed on large MOS set
            seed=opt.seed + i,
        )
        by_dist.setdefault(dist_id, []).append((sz, mos))
        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            print(f"[VAL kadid] {i+1}/{len(rows)}", end="\r")
    print()

    srccs = []
    for dist_id, pairs in sorted(by_dist.items()):
        if len(pairs) < 5:
            continue
        sz = np.asarray([a for a, _ in pairs], dtype=np.float64)
        mos = np.asarray([b for _, b in pairs], dtype=np.float64)
        rho, _ = spearmanr(sz, mos)
        if rho is not None and np.isfinite(rho):
            srccs.append(float(rho))
    if not srccs:
        return out
    out["kadid_srcc"] = float(np.mean(srccs))
    out["kadid_n"] = sum(len(v) for v in by_dist.values())
    out["kadid_n_types"] = len(srccs)
    return out


def update_or_fit_reference(generator, ref_loader, device, opt, mu_ref, Sigma_ref, run_dir):
    mu_new, Sig_new = fit_reference(
        generator, ref_loader, device, max_batches=opt.ref_max_batches
    )
    if mu_ref is None or opt.ref_mode == "every":
        mu_ref, Sigma_ref = mu_new, Sig_new
        print(f"[INFO] fitted S_Z reference (mode={opt.ref_mode})")
    elif opt.ref_mode == "ema":
        mu_ref, Sigma_ref = ema_update_reference(
            mu_ref, Sigma_ref, mu_new, Sig_new, momentum=opt.ref_ema
        )
        print(f"[INFO] EMA-updated S_Z reference (m={opt.ref_ema})")
    else:
        print("[INFO] keeping frozen S_Z reference")
    save_reference(os.path.join(run_dir, "reference_stats.pt"), mu_ref, Sigma_ref)
    return mu_ref, Sigma_ref


# ---------------------------------------------------------------------------
# latent plots (reuse t-SNE style from old code)
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "HR": "r",
    "LR": "g",
    "gblur": "b",
    "jpeg": "m",
    "sharpen": "c",
    "color": "y",
    "pixelate": "orange",
    "other": "k",
}
_TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_latent_2d(Z, labels, title, save_path, method="tsne"):
    fig, ax = plt.subplots(figsize=(7, 6))
    labels = np.asarray(labels)
    uniq = list(np.unique(labels))
    for i, lab in enumerate(uniq):
        idx = labels == lab
        color = LABEL_COLORS.get(lab, _TAB10[i % len(_TAB10)])
        ax.scatter(Z[idx, 0], Z[idx, 1], c=color, label=lab, alpha=0.7, s=18)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    return fig


def compute_latent_embeddings(mus, labels, method, tsne_max, perplexity, seed):
    X = mus
    labs = np.asarray(labels)
    if method == "pca":
        Z = PCA(n_components=2, random_state=seed).fit_transform(X)
        return Z, labs
    if X.shape[0] > tsne_max:
        rng = np.random.RandomState(seed)
        sel = rng.choice(X.shape[0], size=tsne_max, replace=False)
        X = X[sel]
        labs = labs[sel]
    perp = min(perplexity, max(5.0, (X.shape[0] - 1) / 3.0))
    Z = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X)
    return Z, labs


def plot_sz_hist(sz_hr, sz_deg, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(sz_hr, bins=30, alpha=0.6, label="HR", color="r")
    ax.hist(sz_deg, bins=30, alpha=0.6, label="degraded", color="b")
    ax.set_xlabel("S_Z (higher = worse quality)")
    ax.set_ylabel("count")
    ax.set_title("S_Z distribution: HR vs degraded")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    return fig


def latent_plot_methods(opt):
    methods = []
    if opt.latent_plot in ("pca", "both"):
        methods.append("pca")
    if opt.latent_plot in ("tsne", "both"):
        methods.append("tsne")
    return methods


class PathLabelImageDataset(torch.utils.data.Dataset):
    """Center-crop images with explicit labels (pool name or HR/LR/...)."""

    def __init__(self, path_labels, crop_size=256, transform=None):
        self.path_labels = list(path_labels)
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.path_labels)

    def __getitem__(self, idx):
        path, label = self.path_labels[idx]
        basename = os.path.basename(path)
        image = Image.open(path).convert("RGB")
        image = _resize_short_side(image, self.crop_size)
        image = center_crop(image, self.crop_size)
        image = np.asarray(image)
        image = (image / 255.0).astype("float32")
        if self.transform:
            image = self.transform(image)
        return image, basename, label


def subsample_path_labels(path_labels, max_n, seed):
    if max_n is None or len(path_labels) <= max_n:
        return list(path_labels)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(path_labels), size=max_n, replace=False)
    return [path_labels[i] for i in sorted(idx.tolist())]


def cap_train_path_labels(path_pool_labels, max_n, seed):
    if max_n is None or len(path_pool_labels) <= max_n:
        return list(path_pool_labels)
    primary = [x for x in path_pool_labels if x[1] != "KADIS-700k"]
    kadis = [x for x in path_pool_labels if x[1] == "KADIS-700k"]
    if len(primary) >= max_n:
        return subsample_path_labels(primary, max_n, seed)
    need = max_n - len(primary)
    return primary + subsample_path_labels(kadis, need, seed)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    generator,
    eval_loader,
    device,
    mu_ref,
    Sigma_ref,
    opt,
    epoch,
    writer,
    plot_dirs,
    transform_inv,
    transform=None,
    eval_files=None,
):
    from dataloader import _label_from_basename

    generator.eval()

    mus, labels = [], []
    sz_all, is_hr = [], []
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    grid_imgs = []
    recon_pairs = []

    skw = sz_kwargs(opt)
    n_crops = max(1, int(getattr(opt, "n_eval_crops", 1)))

    if n_crops > 1 and eval_files and transform is not None:
        for i, path in enumerate(eval_files):
            bn = os.path.basename(path)
            lab = _label_from_basename(bn)
            sz, mu_m, x0, recon0 = score_image_multicrop(
                generator,
                path,
                opt.img,
                transform,
                device,
                mu_ref,
                Sigma_ref,
                skw,
                n_crops,
                seed=opt.seed + i * 17,
            )
            mus.append(mu_m)
            labels.append(lab)
            is_hr.append(1 if lab == "HR" else 0)
            sz_all.append(torch.tensor([sz]))
            psnr_sum += psnr_batch(recon0, x0) * 1.0
            ssim_sum += ssim_batch(recon0, x0) * 1.0
            n += 1
            if len(grid_imgs) < 4:
                pair = torch.cat([x0[0].cpu(), recon0[0].cpu()], dim=2)
                grid_imgs.append(pair)
                recon_pairs.append(
                    (
                        transform_inv(x0[0].cpu()),
                        transform_inv(recon0[0].cpu().clamp(0, 1)),
                    )
                )
            print(f"[VAL crop] {i+1}/{len(eval_files)} {bn}", end="\r")
        print()
    else:
        for i, (data, basename, label) in enumerate(eval_loader):
            data = data.to(device)
            recon, mu, log_var = generator(data)
            sz = sz_from_stats(mu, log_var, mu_ref, Sigma_ref, **skw)

            mus.append(mu.cpu())
            for lab in label:
                labels.append(lab)
                is_hr.append(1 if lab == "HR" else 0)
            sz_all.append(sz.cpu())

            psnr_sum += psnr_batch(recon, data) * data.size(0)
            ssim_sum += ssim_batch(recon, data) * data.size(0)
            n += data.size(0)

            if len(grid_imgs) < 4:
                for b in range(min(data.size(0), 4 - len(grid_imgs))):
                    pair = torch.cat([data[b].cpu(), recon[b].cpu()], dim=2)
                    grid_imgs.append(pair)
                    if len(recon_pairs) < 4:
                        recon_pairs.append(
                            (
                                transform_inv(data[b].cpu()),
                                transform_inv(recon[b].cpu().clamp(0, 1)),
                            )
                        )

    mus_np = torch.cat(mus, dim=0).numpy()
    sz_np = torch.cat(sz_all, dim=0).numpy()
    is_hr = np.asarray(is_hr)
    labels_arr = np.asarray(labels)

    val_psnr = psnr_sum / max(n, 1)
    val_ssim = ssim_sum / max(n, 1)

    y_true = 1 - is_hr  # 1 = degraded; S_Z higher for degraded => AUROC
    try:
        if len(np.unique(y_true)) > 1:
            auroc = float(roc_auc_score(y_true, sz_np))
        else:
            auroc = float("nan")
    except Exception:
        auroc = float("nan")

    ladder = {
        "blur_rho": float("nan"),
        "jpeg_rho": float("nan"),
        "lr_rho": float("nan"),
        "sharpen_rho": float("nan"),
        "color_rho": float("nan"),
        "pixelate_rho": float("nan"),
        "ladder_spearman": float("nan"),
    }
    if getattr(opt, "ladder_meta", None) and transform is not None:
        print("[VAL] scoring degradation ladders ...")
        ladder = eval_ladder_spearman(
            generator, opt.ladder_meta, transform, device, mu_ref, Sigma_ref, opt
        )
        print(
            f"[VAL] ladder Kendall blur={ladder['blur_rho']:.4f} "
            f"jpeg={ladder['jpeg_rho']:.4f} lr={ladder['lr_rho']:.4f} "
            f"sharpen={ladder.get('sharpen_rho', float('nan')):.4f} "
            f"color={ladder.get('color_rho', float('nan')):.4f} "
            f"pixelate={ladder.get('pixelate_rho', float('nan')):.4f} "
            f"mean={ladder['ladder_spearman']:.4f}"
        )

    kadid = {"kadid_srcc": float("nan"), "kadid_n": 0, "kadid_n_types": 0}
    if getattr(opt, "kadid_monitor_csv", None) and transform is not None:
        print("[VAL] scoring KADID monitor MOS set ...")
        kadid = eval_kadid_monitor(
            generator, opt.kadid_monitor_csv, transform, device, mu_ref, Sigma_ref, opt
        )
        print(
            f"[VAL] kadid_srcc={kadid['kadid_srcc']:.4f} "
            f"(n={kadid['kadid_n']}, types={kadid['kadid_n_types']}; "
            f"more negative = better)"
        )

    epoch_tag = f"{epoch:04d}.png"

    if grid_imgs:
        grid = torch.stack(grid_imgs[:4], dim=0)
        writer.add_images("val/input_recon", grid, epoch, dataformats="NCHW")
    if recon_pairs:
        recon_path = os.path.join(plot_dirs["val_recon"], epoch_tag)
        tiles = []
        for inp, rec in recon_pairs:
            w1, h1 = inp.size
            tile = Image.new("RGB", (w1 + rec.size[0], h1))
            tile.paste(inp, (0, 0))
            tile.paste(rec, (w1, 0))
            tiles.append(tile)
        tw, th = tiles[0].size
        canvas = Image.new("RGB", (tw, th * len(tiles)))
        for ti, tile in enumerate(tiles):
            canvas.paste(tile, (0, ti * th))
        canvas.save(recon_path)
        print(f"[INFO] saved val recon -> {recon_path}")

    for method in latent_plot_methods(opt):
        try:
            Z, labs = compute_latent_embeddings(
                mus_np, labels_arr, method, opt.tsne_max, opt.tsne_perplexity, opt.seed
            )
            png = os.path.join(plot_dirs["latent_plots"], f"{method}_{epoch_tag}")
            fig = plot_latent_2d(
                Z, labs, f"{method.upper()} latent (epoch {epoch})", png, method=method
            )
            writer.add_figure(f"latent/{method}", fig, epoch)
            plt.close(fig)
            print(f"[INFO] saved {method} latent plot -> {png}")
        except Exception as e:
            print(f"[WARN] could not plot {method} for epoch {epoch}: {e}")

    sz_hr = sz_np[is_hr == 1]
    sz_deg = sz_np[is_hr == 0]
    hist_path = os.path.join(plot_dirs["sz_hist"], epoch_tag)
    fig = plot_sz_hist(sz_hr, sz_deg, hist_path)
    writer.add_figure("sz/histogram", fig, epoch)
    plt.close(fig)
    print(f"[INFO] saved S_Z hist -> {hist_path}")

    writer.add_scalar("val/PSNR", val_psnr, epoch)
    writer.add_scalar("val/SSIM", val_ssim, epoch)
    writer.add_scalar("val/SZ_AUROC", auroc, epoch)
    writer.add_scalar("val/SZ_mean_HR", float(sz_hr.mean()) if len(sz_hr) else 0.0, epoch)
    writer.add_scalar("val/SZ_mean_deg", float(sz_deg.mean()) if len(sz_deg) else 0.0, epoch)
    if np.isfinite(ladder["ladder_spearman"]):
        writer.add_scalar("val/ladder_spearman", ladder["ladder_spearman"], epoch)
        writer.add_scalar("val/blur_rho", ladder["blur_rho"], epoch)
        writer.add_scalar("val/jpeg_rho", ladder["jpeg_rho"], epoch)
    if np.isfinite(kadid["kadid_srcc"]):
        writer.add_scalar("val/kadid_srcc", kadid["kadid_srcc"], epoch)

    return {
        "epoch": epoch,
        "psnr": val_psnr,
        "ssim": val_ssim,
        "sz_auroc": auroc,
        "sz_mean_hr": float(sz_hr.mean()) if len(sz_hr) else float("nan"),
        "sz_mean_deg": float(sz_deg.mean()) if len(sz_deg) else float("nan"),
        **ladder,
        **kadid,
    }


def best_score(metrics, best_metric):
    if best_metric == "separation":
        return metrics.get("sz_auroc", float("-inf"))
    if best_metric == "psnr":
        return metrics.get("psnr", float("-inf"))
    if best_metric == "lpips":
        return -metrics.get("lpips", float("inf"))
    if best_metric == "ladder_spearman":
        return metrics.get("ladder_spearman", float("-inf"))
    if best_metric == "kadid_srcc":
        # more negative SRCC is better => invert sign for "higher is better" selection
        v = metrics.get("kadid_srcc", float("nan"))
        if not np.isfinite(v):
            return float("-inf")
        return -float(v)
    return metrics.get("sz_auroc", float("-inf"))


# ---------------------------------------------------------------------------
# checkpoint (CVAE only — single optimizer)
# ---------------------------------------------------------------------------

def save_checkpoint(path, generator, optimizer, epoch, scaler, config, seed, best_extras=None):
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    payload = {
        "generator": generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "seed": seed,
        "rng_state": rng_state,
    }
    if best_extras:
        payload.update(best_extras)
    torch.save(payload, path)


def load_checkpoint(path, generator, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt["epoch"])
    if "rng_state" in ckpt and ckpt["rng_state"] is not None:
        rs = ckpt["rng_state"]
        try:
            random.setstate(rs["python"])
            np.random.set_state(rs["numpy"])
            torch.set_rng_state(rs["torch"])
            if rs.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rs["cuda"])
        except Exception as e:
            print(f"[WARN] could not fully restore rng_state: {e}")
    print(f"[INFO] Training Resumed from epoch {start_epoch}")
    return start_epoch, ckpt


def load_weights_only(path, generator, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator"])
    src_ep = ckpt.get("epoch", "?")
    print(f"[INFO] warm-started generator from {path} (src epoch={src_ep})")
    return ckpt


def build_lr_scheduler(optimizer, opt):
    if opt.lr_schedule == "none":
        return None
    if opt.lr_schedule == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(opt.epochs, 1), eta_min=opt.lr_min
        )
    if opt.lr_schedule == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=max(opt.epochs // 3, 1), gamma=0.5
        )
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    opt = get_args()
    set_seed(opt.seed)

    if not torch.cuda.is_available():
        raise Exception("No GPU found")
    print("CUDA Version:", torch.version.cuda)
    torch.cuda.set_device("cuda:" + str(opt.gpu))
    device = torch.device("cuda:" + str(opt.gpu))
    print(
        "current CUDA Device:",
        torch.cuda.current_device(),
        torch.cuda.get_device_name(torch.cuda.current_device()),
    )

    train_pools = [p.strip() for p in opt.pools.split(",") if p.strip()]
    for p in train_pools:
        if p in HOLD_OUT_POOLS:
            print(f"[WARN] pool '{p}' is normally held out for eval; including anyway as requested")

    file_list = []
    path_pool_labels = []
    per_pool_counts = {}

    if opt.train_list and os.path.isfile(opt.train_list):
        # Preferred path: use an explicit pristine image list
        file_list = read_manifest(opt.train_list)
        train_csv = opt.train_list.replace(".txt", ".csv")
        if os.path.isfile(train_csv):
            with open(train_csv) as f:
                for r in csv.DictReader(f):
                    path_pool_labels.append((r["path"], r["pool"]))
                    per_pool_counts[r["pool"]] = per_pool_counts.get(r["pool"], 0) + 1
            # keep file_list order consistent with csv if lengths match
            if len(path_pool_labels) == len(file_list):
                file_list = [p for p, _ in path_pool_labels]
            else:
                path_pool_labels = [(p, "train_list") for p in file_list]
                per_pool_counts = {"train_list": len(file_list)}
        else:
            path_pool_labels = [(p, "train_list") for p in file_list]
            per_pool_counts = {"train_list": len(file_list)}
        train_pools = sorted(per_pool_counts.keys())
        print(f"[INFO] train_list={opt.train_list} ({len(file_list)} images)")
    else:
        print(
            "[ERROR] Provide --train_list pointing to a text file with one pristine "
            "image path per line.\n"
            "  See examples/train_list_example.txt"
        )
        sys.exit(1)

    if opt.max_train_images is not None:
        before = len(path_pool_labels)
        path_pool_labels = cap_train_path_labels(
            path_pool_labels, opt.max_train_images, opt.seed
        )
        file_list = [p for p, _ in path_pool_labels]
        per_pool_counts = {}
        for _, pool in path_pool_labels:
            per_pool_counts[pool] = per_pool_counts.get(pool, 0) + 1
        print(
            f"[INFO] capped train images: {before} -> {len(file_list)} "
            f"(--max_train_images={opt.max_train_images})"
        )

    run_name = opt.run_name or make_run_name(opt, train_pools)
    output_root = os.path.abspath(opt.output_root)
    ckpt_dir = os.path.join(output_root, "checkpoints", run_name)
    run_dir = os.path.join(output_root, "runs", run_name)
    plot_dirs = {
        "train_recon": os.path.join(run_dir, "train_recon"),
        "val_recon": os.path.join(run_dir, "val_recon"),
        "latent_plots": os.path.join(run_dir, "latent_plots"),
        "sz_hist": os.path.join(run_dir, "sz_hist"),
    }
    cfg_dir = os.path.join(output_root, "configs")

    if run_already_started(ckpt_dir, run_dir) and not opt.resume:
        print(
            f"[ERROR] run '{run_name}' already exists.\n"
            f"  Resume same version:  add --resume  (loads {os.path.join(ckpt_dir, 'last.pth')})\n"
            f"  Start a new run:      bump --ver (e.g. --ver 1.1)"
        )
        sys.exit(1)

    os.makedirs(ckpt_dir, exist_ok=True)
    for d in plot_dirs.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(cfg_dir, exist_ok=True)

    git_hash = get_git_hash()
    config = vars(opt).copy()
    config["run_name"] = run_name
    config["train_pools"] = train_pools
    config["per_pool_counts"] = per_pool_counts
    config["total_train_images"] = len(file_list)
    config["git_commit"] = git_hash
    config["normalization"] = "[0,1]"
    config["crop_size"] = opt.img
    config["model"] = "CVAEGenerator_v2"
    config["losses"] = "L_pix + w_vgg*L_vgg + beta_kl*L_KL (no discriminator)"

    print(f"[INFO] run_name: {run_name}")
    print(f"[INFO] version:  {version_tag(opt.ver)}")
    print(f"[INFO] run_dir:  {os.path.abspath(run_dir)}")
    print(f"[INFO] ckpt_dir: {os.path.abspath(ckpt_dir)}")
    print(f"[INFO] git commit: {git_hash}")
    print(f"[INFO] chosen pools: {train_pools}")
    for p, c in per_pool_counts.items():
        print(f"       {p}: {c}")
    print(f"[INFO] total train images: {len(file_list)}")
    print(f"[INFO] crop size: {opt.img} | norm: [0,1] | ldim: {opt.ldim} | amp: {opt.amp}")
    print(f"[INFO] best_metric: {opt.best_metric} | latent_plot: {opt.latent_plot}")
    print(
        f"[INFO] S_Z: mode={opt.sz_mode} ref_mode={opt.ref_mode} "
        f"lambda_rank={opt.lambda_rank} free_bits={opt.free_bits}"
    )
    print("[INFO] CVAE ONLY (VGG+KLD) — no discriminator / adversarial loss")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(cfg_dir, f"{run_name}.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(run_dir, "train_files.txt"), "w") as f:
        for p in file_list:
            f.write(p + "\n")
    with open(os.path.join(run_dir, "git_commit.txt"), "w") as f:
        f.write(git_hash + "\n")
    with open(os.path.join(run_dir, "seed.txt"), "w") as f:
        f.write(str(opt.seed) + "\n")

    transform = transforms.Compose([transforms.ToTensor()])
    transform_inv = transforms.ToPILImage()

    train_dataset = ManifestImageDataset(
        file_list, crop_size=opt.img, hflip=True, transform=transform
    )
    g = torch.Generator()
    g.manual_seed(opt.seed)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=opt.threads,
        batch_size=opt.batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        drop_last=True,
    )

    if opt.eval_list:
        eval_files = load_eval_list(opt.eval_list)
        print(f"[INFO] eval_list={opt.eval_list} ({len(eval_files)} images)")
    else:
        eval_files = collect_eval_files(opt.eval_roots)
    if not eval_files:
        print(f"[ERROR] no eval images under {opt.eval_roots} / {opt.eval_list}")
        sys.exit(1)
    print(
        f"[INFO] fixed eval set ({opt.eval_pool}): {len(eval_files)} images "
        f"| n_eval_crops={opt.n_eval_crops} | ladder_meta={opt.ladder_meta}"
    )
    with open(os.path.join(run_dir, "eval_files.txt"), "w") as f:
        for p in eval_files:
            f.write(p + "\n")

    eval_dataset = EvalImageDataset(eval_files, crop_size=opt.img, transform=transform)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=min(4, opt.batch_size), shuffle=False, num_workers=2
    )

    # Deterministic center-crop pristine set for S_Z reference (stratified subsample)
    ref_entries = subsample_path_labels(
        path_pool_labels, min(len(path_pool_labels), 2000), opt.seed
    )
    ref_dataset = PathLabelImageDataset(
        ref_entries, crop_size=opt.img, transform=transform
    )
    ref_loader = torch.utils.data.DataLoader(
        ref_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2
    )
    print(f"[INFO] S_Z reference set: {len(ref_entries)} center-crop pristine images")

    # models — ldim IS used (old bug: silent global latent_dim=64)
    generator = CVAEGenerator_v2(latent_dim=opt.ldim, image_size=opt.img).to(device)

    if opt.pix_loss == "l1":
        pix_crit = nn.L1Loss(reduction="mean")
    else:
        pix_crit = nn.MSELoss(reduction="mean")
    vgg_loss_fn = VGGPerceptualLoss(device)

    optimizer = optim.Adam(generator.parameters(), lr=opt.lr, betas=tuple(opt.betas))
    scaler = GradScaler("cuda", enabled=opt.amp)
    sched = build_lr_scheduler(optimizer, opt)

    start_epoch = 0
    global_step = 0
    best_val = float("-inf")
    bad_vals = 0
    w_vgg = opt.w_vgg
    balanced_once = False

    last_path = os.path.join(ckpt_dir, "last.pth")
    best_path = os.path.join(ckpt_dir, "best.pth")
    best_psnr_path = os.path.join(ckpt_dir, "best_psnr.pth")
    best_psnr_val = float("-inf")

    if opt.init_from and not opt.resume:
        if not os.path.isfile(opt.init_from):
            print(f"[ERROR] --init_from missing file: {opt.init_from}")
            sys.exit(1)
        load_weights_only(opt.init_from, generator, device)
        # Skip KL warmup after warm-start (re-ramping beta from 0 collapses the posterior)
        global_step = max(opt.kl_warmup_steps, 0)
        print(
            f"[INFO] warm-start: KL warmup skipped "
            f"(global_step={global_step}, beta_kl={opt.beta_kl})"
        )
        # Fresh best tracking for this run (always write a run-local best.pth)

    if opt.resume:
        if not os.path.isfile(last_path):
            print(f"[ERROR] --resume but missing {last_path}")
            sys.exit(1)
        print("[INFO] Resuming Training ..........................................")
        start_epoch, ckpt = load_checkpoint(
            last_path, generator, optimizer, scaler, device
        )
        global_step = start_epoch * max(len(dataloader), 1)
        if os.path.isfile(best_path):
            try:
                best_ck = torch.load(best_path, map_location="cpu", weights_only=False)
                if best_ck.get("best_value") is not None:
                    best_val = float(best_ck["best_value"])
                    print(
                        f"[INFO] restored best_val={best_val:.4f} "
                        f"(metric={best_ck.get('best_metric', opt.best_metric)}, "
                        f"epoch={best_ck.get('epoch')})"
                    )
            except Exception as e:
                print(f"[WARN] could not restore best_val from best.pth: {e}")
        if os.path.isfile(best_psnr_path):
            try:
                best_psnr_ck = torch.load(best_psnr_path, map_location="cpu", weights_only=False)
                if best_psnr_ck.get("val_metrics", {}).get("psnr") is not None:
                    best_psnr_val = float(best_psnr_ck["val_metrics"]["psnr"])
                    print(
                        f"[INFO] restored best_psnr_val={best_psnr_val:.4f} "
                        f"(epoch={best_psnr_ck.get('epoch')})"
                    )
            except Exception as e:
                print(f"[WARN] could not restore best_psnr_val from best_psnr.pth: {e}")
        balanced_once = True
        if start_epoch >= opt.epochs:
            print(
                f"[ERROR] checkpoint epoch {start_epoch} >= --epochs {opt.epochs}. "
                f"Increase --epochs to continue (e.g. --epochs {start_epoch + 100})."
            )
            sys.exit(1)

    # Fit / restore S_Z reference (frozen under ref_mode=once)
    ref_pt = os.path.join(run_dir, "reference_stats.pt")
    if opt.resume and opt.ref_mode == "once" and os.path.isfile(ref_pt):
        mu_ref, Sigma_ref = load_reference(ref_pt)
        print(f"[INFO] restored frozen S_Z reference from {ref_pt}")
    elif opt.init_ref_from and not opt.resume and opt.ref_mode == "once":
        if not os.path.isfile(opt.init_ref_from):
            print(f"[ERROR] --init_ref_from missing file: {opt.init_ref_from}")
            sys.exit(1)
        mu_ref, Sigma_ref = load_reference(opt.init_ref_from)
        save_reference(ref_pt, mu_ref, Sigma_ref)
        print(
            f"[INFO] loaded frozen S_Z reference from --init_ref_from={opt.init_ref_from} "
            f"(overrides any mu_ref/Sigma_ref embedded in --init_from)"
        )
    elif (
        opt.init_from
        and not opt.resume
        and opt.ref_mode == "once"
        and os.path.isfile(opt.init_from)
    ):
        # Prefer reference stats already stored in the warm-start checkpoint
        init_ck = torch.load(opt.init_from, map_location="cpu", weights_only=False)
        if "mu_ref" in init_ck and "Sigma_ref" in init_ck:
            mu_ref = init_ck["mu_ref"].cpu()
            Sigma_ref = init_ck["Sigma_ref"].cpu()
            save_reference(ref_pt, mu_ref, Sigma_ref)
            print(
                f"[INFO] loaded frozen S_Z reference from {opt.init_from} "
                f"(ref_mode=once; not refitting)"
            )
        else:
            mu_ref, Sigma_ref = fit_reference(
                generator, ref_loader, device, max_batches=opt.ref_max_batches
            )
            save_reference(ref_pt, mu_ref, Sigma_ref)
            print("[INFO] init checkpoint had no mu_ref/Sigma_ref; fitted a new reference")
    else:
        mu_ref, Sigma_ref = fit_reference(
            generator, ref_loader, device, max_batches=opt.ref_max_batches
        )
        save_reference(ref_pt, mu_ref, Sigma_ref)
        print(
            f"[INFO] initial S_Z reference ready (mode={opt.ref_mode}, "
            f"sz_mode={opt.sz_mode}, lambda_rank={opt.lambda_rank})"
        )

    writer = SummaryWriter(log_dir=run_dir)
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "l_total",
                    "l_pix",
                    "l_vgg",
                    "l_kl",
                    "l_rank",
                    "psnr",
                    "ssim",
                    "sz_auroc",
                    "grad_norm_pix",
                    "grad_norm_vgg",
                    "grad_norm_kl",
                    "w_vgg",
                    "beta_kl_eff",
                ]
            )

    for epoch in range(start_epoch, opt.epochs):
        generator.train()

        sum_tot = sum_pix = sum_vgg = sum_kl = sum_rank = 0.0
        n_batches = 0
        last_gn_pix = last_gn_vgg = last_gn_kl = float("nan")
        last_beta = opt.beta_kl
        skw = sz_kwargs(opt)
        mu_ref_d = mu_ref.to(device)
        Sigma_ref_d = Sigma_ref.to(device)

        st_time = time.time()
        for i, data in enumerate(dataloader):
            it_time = time.time()
            real_images = data.to(device)

            optimizer.zero_grad(set_to_none=True)
            beta_eff = kl_warmup_beta(global_step, opt.kl_warmup_steps, opt.beta_kl)
            last_beta = beta_eff

            with autocast("cuda", enabled=opt.amp):
                recon, mu, log_var = generator(real_images)
                L_pix = pix_crit(recon, real_images)
                L_vgg = vgg_loss_fn(recon, real_images)
                L_KL_raw = kl_loss(mu, log_var)
                if opt.free_bits > 0:
                    L_KL = torch.clamp(L_KL_raw, min=opt.free_bits)
                else:
                    L_KL = L_KL_raw
                L_total = opt.w_pix * L_pix + w_vgg * L_vgg + beta_eff * L_KL

                L_rank = torch.zeros((), device=device)
                if opt.lambda_rank > 0 and mu_ref is not None:
                    deg = synthesize_degraded(real_images)
                    _, mu_d, lv_d = generator(deg)
                    sz_c = sz_from_stats(mu, log_var, mu_ref_d, Sigma_ref_d, **skw)
                    sz_d = sz_from_stats(mu_d, lv_d, mu_ref_d, Sigma_ref_d, **skw)
                    L_rank = ranking_sz_loss(sz_c, sz_d, opt.rank_margin)
                    L_total = L_total + opt.lambda_rank * L_rank

            scaler.scale(L_total).backward()
            scaler.step(optimizer)
            scaler.update()

            # periodic per-term gradient norms (no amp)
            if (global_step % opt.grad_log_every == 0) and global_step > 0:
                generator.zero_grad(set_to_none=True)
                recon_i, mu_i, lv_i = generator(real_images)
                lp = pix_crit(recon_i, real_images)
                lp.backward(retain_graph=True)
                last_gn_pix = grad_norm(generator.parameters())
                generator.zero_grad(set_to_none=True)

                recon_i, mu_i, lv_i = generator(real_images)
                lv = vgg_loss_fn(recon_i, real_images)
                lv.backward(retain_graph=True)
                last_gn_vgg = grad_norm(generator.parameters())
                generator.zero_grad(set_to_none=True)

                recon_i, mu_i, lv_i = generator(real_images)
                lk = kl_loss(mu_i, lv_i)
                lk.backward()
                last_gn_kl = grad_norm(generator.parameters())
                generator.zero_grad(set_to_none=True)

                if opt.auto_balance and not balanced_once and last_gn_pix > 0:
                    if last_gn_vgg > 1e-8:
                        w_vgg = float(
                            np.clip(
                                last_gn_pix / last_gn_vgg,
                                opt.balance_w_vgg_min,
                                opt.balance_w_vgg_max,
                            )
                        )
                    balanced_once = True
                    print(
                        f"\n[INFO] auto-balance: w_vgg={w_vgg:.4f} "
                        f"(gn_pix={last_gn_pix:.3f} gn_vgg={last_gn_vgg:.3f} gn_kl={last_gn_kl:.3f})"
                    )

            # occasional train recon PNG
            if opt.train_recon_every > 0 and (global_step % opt.train_recon_every == 0):
                with torch.no_grad():
                    inp_pil = transform_inv(real_images[0].cpu())
                    rec_pil = transform_inv(recon[0].detach().cpu().float().clamp(0, 1))
                    outp = os.path.join(
                        plot_dirs["train_recon"],
                        f"e{epoch+1:04d}_s{global_step:06d}.png",
                    )
                    save_intermediateResults(inp_pil, rec_pil, outp)

            sum_tot += L_total.item()
            sum_pix += L_pix.item()
            sum_vgg += L_vgg.item()
            sum_kl += L_KL_raw.item()
            sum_rank += float(L_rank.item()) if torch.is_tensor(L_rank) else 0.0
            n_batches += 1
            global_step += 1

            print(
                f"Epoch {epoch+1}/{opt.epochs} iter {i}/{len(dataloader)} "
                f"L_total: {L_total.item():.4f} L_pix: {L_pix.item():.4f} "
                f"L_vgg: {L_vgg.item():.4f} L_KL: {L_KL_raw.item():.4f} "
                f"L_rank: {float(L_rank.item()):.4f} "
                f"Time: {time.time() - it_time:.2f}",
                end="\r",
            )

        ep = epoch + 1
        AvgTot = sum_tot / max(n_batches, 1)
        AvgPix = sum_pix / max(n_batches, 1)
        AvgVgg = sum_vgg / max(n_batches, 1)
        AvgKL = sum_kl / max(n_batches, 1)
        AvgRank = sum_rank / max(n_batches, 1)

        writer.add_scalar("train/L_total", AvgTot, ep)
        writer.add_scalar("train/L_pix", AvgPix, ep)
        writer.add_scalar("train/L_vgg", AvgVgg, ep)
        writer.add_scalar("train/L_KL", AvgKL, ep)
        writer.add_scalar("train/L_rank", AvgRank, ep)
        writer.add_scalar("train/beta_kl_eff", last_beta, ep)
        writer.add_scalar("train/w_vgg", w_vgg, ep)
        writer.add_scalar("train/LR", optimizer.param_groups[0]["lr"], ep)
        if not np.isnan(last_gn_pix):
            writer.add_scalar("grad/norm_pix", last_gn_pix, ep)
        if not np.isnan(last_gn_vgg):
            writer.add_scalar("grad/norm_vgg", last_gn_vgg, ep)
        if not np.isnan(last_gn_kl):
            writer.add_scalar("grad/norm_kl", last_gn_kl, ep)

        print(
            f"Epoch {ep}/{opt.epochs} L_total: {AvgTot:.4f} L_pix: {AvgPix:.4f} "
            f"L_vgg: {AvgVgg:.4f} L_KL: {AvgKL:.4f} L_rank: {AvgRank:.4f} "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} "
            f"Time: {time.time() - st_time:.2f}"
        )
        if AvgKL < 1e-4:
            print(f"[WARN] KL near-collapsed (mean L_KL={AvgKL:.6f}) — check free_bits / warmup")
        if AvgKL > 100:
            print(f"[WARN] KL exploding (mean L_KL={AvgKL:.2f})")

        save_checkpoint(
            last_path,
            generator,
            optimizer,
            epoch + 1,
            scaler if opt.amp else None,
            config,
            opt.seed,
        )
        if opt.save_epoch_every > 0 and (
            ((epoch + 1) % opt.save_epoch_every == 0) or (epoch + 1 == opt.epochs)
        ):
            epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1:04d}.pth")
            save_checkpoint(
                epoch_path,
                generator,
                optimizer,
                epoch + 1,
                scaler if opt.amp else None,
                config,
                opt.seed,
                best_extras={
                    "mu_ref": mu_ref.cpu(),
                    "Sigma_ref": Sigma_ref.cpu(),
                    "checkpoint_kind": "periodic_epoch",
                    "sz_mode": opt.sz_mode,
                    "sz_sigma_t_max": opt.sz_sigma_t_max,
                },
            )
            print(f"[INFO] saved periodic checkpoint -> {epoch_path}")

        row = [
            epoch + 1,
            AvgTot,
            AvgPix,
            AvgVgg,
            AvgKL,
            AvgRank,
            "",
            "",
            "",
            last_gn_pix,
            last_gn_vgg,
            last_gn_kl,
            w_vgg,
            last_beta,
        ]

        stop_early = False
        if ((epoch + 1) % opt.val_every == 0) or (epoch + 1 == opt.epochs):
            if opt.ref_mode != "once":
                mu_ref, Sigma_ref = update_or_fit_reference(
                    generator, ref_loader, device, opt, mu_ref, Sigma_ref, run_dir
                )
            else:
                save_reference(
                    os.path.join(run_dir, "reference_stats.pt"), mu_ref, Sigma_ref
                )

            metrics = run_validation(
                generator,
                eval_loader,
                device,
                mu_ref,
                Sigma_ref,
                opt,
                epoch + 1,
                writer,
                plot_dirs,
                transform_inv,
                transform=transform,
                eval_files=eval_files,
            )
            row[6] = metrics["psnr"]
            row[7] = metrics["ssim"]
            row[8] = metrics["sz_auroc"]

            print(
                f"[VAL] epoch {epoch+1} PSNR={metrics['psnr']:.3f} SSIM={metrics['ssim']:.4f} "
                f"S_Z AUROC={metrics['sz_auroc']:.4f} "
                f"ladder={metrics.get('ladder_spearman', float('nan')):.4f} "
                f"kadid_srcc={metrics.get('kadid_srcc', float('nan')):.4f} "
                f"(HR mean S_Z={metrics['sz_mean_hr']:.3f}, deg={metrics['sz_mean_deg']:.3f})"
            )

            score = best_score(metrics, opt.best_metric)
            if metrics.get("psnr") is not None and not np.isnan(metrics["psnr"]) and metrics["psnr"] > best_psnr_val:
                best_psnr_val = float(metrics["psnr"])
                save_checkpoint(
                    best_psnr_path,
                    generator,
                    optimizer,
                    epoch + 1,
                    scaler if opt.amp else None,
                    config,
                    opt.seed,
                    best_extras={
                        "mu_ref": mu_ref.cpu(),
                        "Sigma_ref": Sigma_ref.cpu(),
                        "best_metric": "psnr",
                        "best_value": best_psnr_val,
                        "val_metrics": metrics,
                        "sz_mode": opt.sz_mode,
                        "sz_sigma_t_max": opt.sz_sigma_t_max,
                    },
                )
                print(f"[INFO] best_psnr.pth improved (psnr={best_psnr_val:.4f}) -> {best_psnr_path}")
            if score > best_val and not np.isnan(score):
                best_val = score
                bad_vals = 0
                save_checkpoint(
                    best_path,
                    generator,
                    optimizer,
                    epoch + 1,
                    scaler if opt.amp else None,
                    config,
                    opt.seed,
                    best_extras={
                        "mu_ref": mu_ref.cpu(),
                        "Sigma_ref": Sigma_ref.cpu(),
                        "best_metric": opt.best_metric,
                        "best_value": best_val,
                        "val_metrics": metrics,
                        "sz_mode": opt.sz_mode,
                        "sz_sigma_t_max": opt.sz_sigma_t_max,
                    },
                )
                print(
                    f"[INFO] best.pth improved ({opt.best_metric}={best_val:.4f}) -> {best_path}"
                )
            else:
                bad_vals += 1
                if opt.early_stop_patience > 0 and bad_vals >= opt.early_stop_patience:
                    print(
                        f"[INFO] early stopping: no {opt.best_metric} improvement for "
                        f"{bad_vals} validations (patience={opt.early_stop_patience})"
                    )
                    stop_early = True

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

        if sched is not None:
            sched.step()

        if stop_early:
            break

    writer.close()
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
