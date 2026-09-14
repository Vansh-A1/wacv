#!/usr/bin/env python3
"""Build verified Task 2 references; fail closed when the original pool is absent."""

import argparse
import csv
import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from external.dataloader import _resize_short_side, center_crop
from external.model import CVAEGenerator_v2
from score import sz_from_stats
from reference_math import DISTANCES, EPS, aggregate_posteriors, diagonal_distances

EA_CHECKPOINT = ROOT / "dataset_model2/checkpoints/wacv_ea_day7_seed42/epoch_0005.pth"
EH_CHECKPOINT = ROOT / "checkpoints/hr_combined_ft1/best.pth"
EH_ANCESTOR = ROOT / "checkpoints/best.pth"
EA_MANIFEST = ROOT / "dataset_model2/day5_debug_eval/day7/day7_taskA/day7_reference_paths_used.csv"
SPLIT_MANIFEST = ROOT / "dataset_model2/ref_splits_seed42/combined_kadid_tid_koniq_split_seed42.csv"
EXPECTED_EA_COUNTS = {"KADID-10k": 876, "TID2013": 255, "KONIQ-10k": 869}
LOG = logging.getLogger("task2")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0])
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def canonical(path):
    old = "/data/projectwork/swati_mam/kadid10k/"
    new = "/data/projectwork/swati_mam/new model data/kadid10k/"
    if path.startswith(old):
        path = new + path[len(old):]
    return str(Path(path).resolve())


def validate_reference_rows(rows, split_rows):
    """EA rows must match the authoritative training split, never basename-only."""
    by_path = {}
    forbidden = set()
    for r in split_rows:
        p = canonical(r["distorted_path"])
        if p in by_path:
            raise ValueError(f"Duplicate image in split manifest: {p}")
        by_path[p] = r
        if r["split"] != "train":
            forbidden.add(p)
            if r.get("ref_path") and r["ref_path"] != "NA":
                forbidden.add(canonical(r["ref_path"]))
    result = []
    for i, r in enumerate(rows):
        p = canonical(r["path"])
        match = by_path.get(p)
        if p in forbidden or match is None or match["split"] != "train":
            raise ValueError(f"Non-training or unmatched EA reference: {p}")
        if r["pool"] != match["dataset"]:
            raise ValueError(f"Dataset mismatch: {p}")
        result.append({
            "index": i, "reference_id": f"{match['dataset']}:{match['image_id']}",
            "image_id": match["image_id"], "dataset": match["dataset"],
            "split": "train", "original_path": r["path"], "path": p,
        })
    for field in ("reference_id", "path"):
        if len({r[field] for r in result}) != len(result):
            raise ValueError(f"Duplicate reference {field}")
    if dict(Counter(r["dataset"] for r in result)) != EXPECTED_EA_COUNTS:
        raise ValueError("EA original 876/255/869 reference counts do not match")
    return result


class ReferenceDataset(Dataset):
    def __init__(self, rows, image_size):
        self.rows, self.image_size = rows, image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]["path"]) as source:
            im = _resize_short_side(source.convert("RGB"), self.image_size)
            im = center_crop(im, self.image_size)
            array = (np.asarray(im) / 255.0).astype(np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if tensor.shape != (3, self.image_size, self.image_size) or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid image tensor: {self.rows[i]['path']}")
        return tensor


def posterior(model, x):
    # Identical encoder path to CVAEGenerator_v2.forward; skip only the decoder.
    for layer in (model.enc1, model.enc2, model.enc3, model.enc4, model.enc5, model.enc6):
        x = F.relu(layer(x))
    hidden = model.fc1(F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1))
    return model.fc_mu(hidden), model.fc_log_var(hidden)


def load_checkpoint(path, epoch):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if int(ck["epoch"]) != epoch:
        raise ValueError(f"Wrong stored epoch in {path}: {ck['epoch']}")
    for key in ("mu_ref", "Sigma_ref"):
        t = ck[key]
        if not torch.isfinite(t).all():
            raise ValueError(f"Nonfinite legacy {key}")
    if ck["mu_ref"].shape != ck["Sigma_ref"].shape or ck["mu_ref"].ndim != 1:
        raise ValueError("Unexpected legacy reference geometry")
    return ck


def legacy_export(ck, directory):
    np.savez_compressed(directory / "legacy_reference.npz",
                        mu_ref=ck["mu_ref"].numpy(), Sigma_ref=ck["Sigma_ref"].numpy())
    with np.load(directory / "legacy_reference.npz", allow_pickle=False) as saved:
        for key in ("mu_ref", "Sigma_ref"):
            original = ck[key].numpy()
            assert original.dtype == saved[key].dtype
            assert np.array_equal(original, saved[key])


def variance_only_probe(dim):
    mu = torch.zeros(2, dim, dtype=torch.float64)
    lv = torch.stack([torch.zeros(dim), torch.full((dim,), np.log(4.0))]).double()
    scores = diagonal_distances(mu, lv, torch.zeros(dim), torch.ones(dim))
    return {name: {"matched_variance": float(value[0]), "changed_variance": float(value[1]),
                   "passed": bool(value[1] > value[0] + 1e-10)}
            for name, value in scores.items()}


def encoder_preflight(model, ck, dataset, agg, device, pole):
    x = torch.stack([dataset[i] for i in range(min(3, len(dataset)))]).to(device)
    with torch.no_grad():
        mu, lv = posterior(model, x)
        _, mu_full, lv_full = model(x)
        torch.testing.assert_close(mu, mu_full, rtol=0, atol=0)
        torch.testing.assert_close(lv, lv_full, rtol=0, atol=0)
    rm = torch.from_numpy(agg["mu_agg"]).to(device)
    rv = torch.from_numpy(agg["v_agg"]).to(device)
    targets = {"enc1.weight": model.enc1.weight, "fc_mu.weight": model.fc_mu.weight,
               "fc_log_var.weight": model.fc_log_var.weight}
    rows = []
    for name in DISTANCES:
        mu, lv = posterior(model, x)
        value = diagonal_distances(mu, lv, rm, rv)[name].mean()
        gradients = torch.autograd.grad(value, tuple(targets.values()))
        for (parameter, _), grad in zip(targets.items(), gradients):
            finite, nonzero = bool(torch.isfinite(grad).all()), bool(torch.count_nonzero(grad))
            rows.append({"pole": pole, "distance": name, "parameter": parameter,
                         "n_reference_images": len(x), "mean_distance": value.item(),
                         "gradient_l2": grad.double().norm().item(),
                         "finite": finite, "nonzero": nonzero, "passed": finite and nonzero})
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), ck["generator"][key]), f"Model changed: {key}"
    if not all(r["passed"] for r in rows):
        raise ValueError("Encoder-gradient preflight failed")
    return rows


def build_ea(out, args):
    directory = out / "EA"
    directory.mkdir(exist_ok=True)
    ck = load_checkpoint(EA_CHECKPOINT, 5)
    source_paths = [EA_CHECKPOINT, EA_MANIFEST, SPLIT_MANIFEST, Path(__file__),
                    HERE / "reference_math.py", HERE / "ASSUMPTIONS.md",
                    ROOT / "external/model.py", ROOT / "external/dataloader.py", ROOT / "score.py"]
    hashes = {str(p): digest(p) for p in source_paths}
    rows = validate_reference_rows(read_csv(EA_MANIFEST), read_csv(SPLIT_MANIFEST))
    LOG.info("EA original pool verified: %s", Counter(r["dataset"] for r in rows))
    for r in rows:
        p = Path(r["path"])
        r.update(image_sha256=digest(p), bytes=p.stat().st_size)
    identity = json_digest({"source_hashes": hashes, "rows": rows,
                            "batch_size": args.batch_size, "precision": "float32_no_amp"})
    cached_manifest = directory / "manifest.json"
    cached_posteriors = directory / "reference_posteriors.npz"
    cached = False
    if cached_manifest.exists() and cached_posteriors.exists():
        previous = json.loads(cached_manifest.read_text())
        if previous.get("input_fingerprint") != identity:
            raise ValueError("Existing EA output has different inputs/code; use a new --output directory")
        if previous.get("posterior_sha256") != digest(cached_posteriors):
            raise ValueError("Existing posterior cache checksum mismatch")
        cached = True
    write_csv(directory / "reference_images.csv", rows)
    legacy_export(ck, directory)
    cfg = ck["config"]
    image_size, dim = int(cfg["img"]), int(cfg["ldim"])
    assert image_size == 256 and dim == 100 and ck["mu_ref"].numel() == dim
    model = CVAEGenerator_v2(latent_dim=dim, image_size=image_size)
    model.load_state_dict(ck["generator"], strict=True)
    model.to(args.device).eval()
    dataset = ReferenceDataset(rows, image_size)
    if cached:
        with np.load(cached_posteriors, allow_pickle=False) as p:
            mu, lv = p["mu"], p["logvar"]
            assert list(p["reference_id"]) == [r["reference_id"] for r in rows]
        LOG.info("Reusing checksum-verified EA posterior cache")
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
        means, logvars = [], []
        with torch.no_grad():
            for batch_i, x in enumerate(loader):
                m, v = posterior(model, x.to(args.device, non_blocking=True))
                means.append(m.double().cpu().numpy())
                logvars.append(v.double().cpu().numpy())
                if batch_i % 20 == 0 or batch_i + 1 == len(loader):
                    LOG.info("EA encoding %d/%d images", min((batch_i + 1) * args.batch_size, len(rows)), len(rows))
        mu, lv = np.concatenate(means), np.concatenate(logvars)
    agg = aggregate_posteriors(mu, lv)
    assert mu.shape == lv.shape == (2000, 100)
    np.testing.assert_array_equal(agg["v_aggregate_raw"], agg["v_between"] + agg["v_within"])
    if not cached:
        np.savez_compressed(cached_posteriors, mu=mu, logvar=lv, variance=np.exp(lv),
                            reference_id=np.array([r["reference_id"] for r in rows]),
                            path=np.array([r["path"] for r in rows]),
                            checkpoint_sha256=np.array(hashes[str(EA_CHECKPOINT)]),
                            stored_epoch=np.array(ck["epoch"]))
    np.savez_compressed(directory / "aggregate_reference.npz", **agg)
    values = diagonal_distances(torch.from_numpy(mu), torch.from_numpy(lv),
                                torch.from_numpy(agg["mu_agg"]), torch.from_numpy(agg["v_agg"]))
    values["D0_legacy"] = sz_from_stats(torch.from_numpy(mu).float(), torch.from_numpy(lv).float(),
                                       ck["mu_ref"], ck["Sigma_ref"], mu_only=True,
                                       eps=EPS, sigma_t_max=1.0)
    probe = variance_only_probe(dim)
    distances = []
    for name, scores in values.items():
        array = scores.detach().numpy()
        finite = bool(np.isfinite(array).all())
        variable = bool(np.ptp(array) > 1e-12)
        sensitive = probe[name]["passed"] if name in probe else None
        distances.append({"pole": "EA", "distance": name, "split": "reference_train",
                          "n_images": len(rows), "minimum": float(array.min()),
                          "maximum": float(array.max()), "mean": float(array.mean()),
                          "std_ddof0": float(array.std(ddof=0)), "finite": finite,
                          "variable": variable, "variance_only_sensitive": sensitive,
                          "passed": finite and variable and (sensitive is not False)})
    assert all(r["passed"] for r in distances)
    score_rows = [{"reference_id": r["reference_id"], **{n: float(v[i]) for n, v in values.items()}}
                  for i, r in enumerate(rows)]
    write_csv(directory / "reference_distance_scores.csv", score_rows)
    gradients = encoder_preflight(model, ck, dataset, agg, args.device, "EA")
    old_mu, old_v = ck["mu_ref"].double().numpy(), ck["Sigma_ref"].double().numpy()
    comparisons = [{"pole": "EA", "dimension": i, "legacy_mean": float(old_mu[i]),
                    "aggregate_mean": float(agg["mu_agg"][i]),
                    "mean_difference": float(agg["mu_agg"][i] - old_mu[i]),
                    "legacy_variance": float(old_v[i]), "between_ddof0": float(agg["v_between"][i]),
                    "within_mean_posterior_variance": float(agg["v_within"][i]),
                    "aggregate_variance_raw": float(agg["v_aggregate_raw"][i]),
                    "aggregate_variance_clamped": float(agg["v_agg"][i]),
                    "aggregate_over_legacy_variance": float(agg["v_agg"][i] / old_v[i])}
                   for i in range(dim)]
    for path, expected in hashes.items():
        assert digest(path) == expected, f"Source changed during run: {path}"
    manifest = {
        "status": "COMPLETE", "pole": "EA", "checkpoint": str(EA_CHECKPOINT),
        "stored_epoch": ck["epoch"], "checkpoint_role": "Day7_pre_Day11_ranking_initialization",
        "reference_manifest": str(EA_MANIFEST), "reference_count": len(rows),
        "dataset_counts": dict(Counter(r["dataset"] for r in rows)),
        "input_fingerprint": identity, "source_hashes": hashes,
        "posterior_sha256": digest(cached_posteriors),
        "aggregate_sha256": digest(directory / "aggregate_reference.npz"),
        "legacy_sha256": digest(directory / "legacy_reference.npz"),
        "image_manifest_sha256": digest(directory / "reference_images.csv"),
        "image_size": image_size, "latent_dim": dim, "device": args.device,
        "gpu_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "batch_size": args.batch_size, "amp": False, "aggregate_dtype": "float64",
        "ddof": 0, "variance_clamp": EPS, "variance_upper_cap": None,
        "variance_only_probes": probe, "validation_holdout_overlap": 0,
        "posterior_forward_equivalence": True, "encoder_weights_unchanged": True,
        "legacy_tensors_exactly_preserved": True, "historical_refit_reproduced": False,
        "no_mos_used": True, "training_performed": False,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(cached_manifest, manifest)
    LOG.info("EA complete: finite distances, variance sensitivity and all 9 encoder-gradient checks passed")
    return comparisons, distances, gradients


def document_eh_blocker(out):
    directory = out / "EH"
    directory.mkdir(exist_ok=True)
    ck = load_checkpoint(EH_CHECKPOINT, 20)
    ancestor = load_checkpoint(EH_ANCESTOR, 190)
    same = {k: bool(torch.equal(ck[k], ancestor[k])) for k in ("mu_ref", "Sigma_ref")}
    assert all(same.values()), "EH ancestry evidence changed; investigate before proceeding"
    legacy_export(ck, directory)
    original_list = ROOT / ancestor["config"]["train_list"]
    original_run = ROOT / "runs" / ancestor["config"]["run_name"] / "train_files.txt"
    manifest = {
        "status": "BLOCKED_MISSING_ORIGINAL_REFERENCE_MANIFEST", "pole": "EH",
        "checkpoint": str(EH_CHECKPOINT), "checkpoint_sha256": digest(EH_CHECKPOINT),
        "stored_epoch": ck["epoch"], "checkpoint_role": "pre_Day11b_initialization",
        "ancestor_checkpoint": str(EH_ANCESTOR), "ancestor_sha256": digest(EH_ANCESTOR),
        "ancestor_stored_epoch": ancestor["epoch"], "ancestor_reference_tensor_equality": same,
        "legacy_sha256": digest(directory / "legacy_reference.npz"),
        "legacy_tensors_exactly_preserved": True,
        "ancestor_training_manifest": str(original_list),
        "ancestor_training_manifest_exists": original_list.exists(),
        "ancestor_saved_training_list": str(original_run),
        "ancestor_saved_training_list_exists": original_run.exists(),
        "ancestor_config_training_count_NOT_reference_count": ancestor["config"]["total_train_images"],
        "ancestor_config_training_pools_NOT_reference_counts": ancestor["config"]["per_pool_counts"],
        "later_training_manifest_NOT_original_reference_pool": str(ROOT / "runs/hr_combined_ft1/train_files.txt"),
        "restore_log": str(ROOT / "runs/hr_combined_ft1/background_train.log"),
        "restore_log_sha256": digest(ROOT / "runs/hr_combined_ft1/background_train.log"),
        "reference_count": None, "validation_holdout_exclusion_verified": False,
        "reference_images_encoded": 0, "aggregate_reference_created": False,
        "needed": "Exact original pristine-reference image manifest (or original ordered training list plus original reference sampling provenance), with image files and verifiable split exclusion.",
    }
    write_json(directory / "manifest.json", manifest)
    LOG.warning("EH blocked: inherited legacy tensors are available, but the exact original pristine reference-image list is not")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "reference_stats")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size or worker count")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / "run.log")])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback. Run run_task2.sh on the GPU host.")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    eh = document_eh_blocker(args.output)
    comparisons, distances, gradients = build_ea(args.output, args)
    write_csv(args.output / "reference_comparison.csv", comparisons)
    write_csv(args.output / "distance_preflight.csv", distances)
    write_csv(args.output / "gradient_preflight.csv", gradients)
    definitions = {
        "D0": "Unchanged score.sz_from_stats(mu_only=True, eps=1e-8, sigma_t_max=1.0) and original stored tensors",
        "KL_q_to_reference": "0.5 * sum(log(vr/vx) + (vx + (mu_x-mu_r)^2)/vr - 1)",
        "W2_squared": "sum((mu_x-mu_r)^2 + (sqrt(vx)-sqrt(vr))^2)",
        "Bhattacharyya": "sum((mu_x-mu_r)^2 / ((vx+vr)/2))/8 + sum(log((vx+vr)/2) - (log(vx)+log(vr))/2)/2",
        "aggregate": "mean(mu); var(mu,ddof=0) + mean(exp(logvar))",
        "vx": "max(exp(logvar),1e-8), float64, no upper cap",
        "vr": "max(v_aggregate_raw,1e-8), float64",
        "dimensions": "Diagonal covariance; all distances sum over 100 dimensions",
    }
    write_json(args.output / "distance_definitions.json", definitions)
    write_json(args.output / "task2_status.json", {
        "status": "PARTIAL", "EA": "COMPLETE", "EH": eh["status"],
        "training_performed": False, "needed_for_EH": eh["needed"],
    })
    report = [
        "DAY 12 TASK 2: PARTIAL (EA complete; EH original pool unavailable)",
        "",
        "EA: 2,000 original artifact-reference images, 876 KADID + 255 TID + 869 KonIQ.",
        "All 2,000 match the locked training split; no validation/holdout images used.",
        "Encoder: Day-7 epoch 5, not the ranked Day-11 encoder.",
        "Preserved original legacy tensors exactly; computed posterior-aware aggregate reference.",
        "Population between variance + mean posterior variance; lower clamp 1e-8.",
        "KL(image posterior || reference), squared W2, Bhattacharyya all passed:",
        "finite/variable reference-image scores, variance-only sensitivity, and nonzero",
        "finite gradients into enc1, fc_mu and fc_log_var. No optimizer or training.",
        "",
        "EH: Original stored tensors copied exactly; aggregate NOT generated.",
        "EH legacy tensors exactly equal those of ancestor checkpoints/best.pth (epoch 190).",
        "The later hr_combined run restored/inherited these tensors; its candidate",
        "2,000-image loader does not prove the original reference-image identities.",
        f"Missing ancestor manifest: {eh['ancestor_training_manifest']}",
        f"Missing ancestor run list: {eh['ancestor_saved_training_list']}",
        "The later 19,300-image HR training list is not substituted.",
        "Need the original pristine-reference image list and verified split exclusions.",
        "No EH aggregate, reference posteriors or gradient results are fabricated.",
        "",
        "CSV comparison/preflight rows cover EA only; they do not establish EH completion.",
        "This is a reference-correction/preflight task, not an IQA improvement result.",
        "See ASSUMPTIONS.md for protocol decisions, and EA/EH manifest.json for provenance.",
    ]
    (args.output / "reference_report.txt").write_text("\n".join(report) + "\n")
    LOG.warning("Task 2 PARTIAL. EA complete; EH requires the original pristine-reference manifest.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
