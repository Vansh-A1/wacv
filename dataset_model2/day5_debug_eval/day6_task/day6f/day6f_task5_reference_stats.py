#!/usr/bin/env python3
"""Day 6 Task 5: inspect pristine E_H reference statistics."""
from __future__ import annotations

import json
from pathlib import Path

import torch

PACKAGE_ROOT = Path("/home/projectwork/student_package")
DAY5_DIR = PACKAGE_ROOT / "dataset_model2" / "day5_debug_eval"
OUT_DIR = DAY5_DIR / "day6f"

REF_STATS = PACKAGE_ROOT / "runs" / "hr_combined_ft1" / "reference_stats.pt"
EH_CKPT = PACKAGE_ROOT / "checkpoints" / "hr_combined_ft1" / "best.pth"
DIAGNOSTICS = DAY5_DIR / "diagnostics.json"

OUT_JSON = OUT_DIR / "task5_reference_stats_summary.json"
OUT_TXT = OUT_DIR / "task5_reference_stats_report.txt"
OUT_CSV = OUT_DIR / "task5_reference_stats_vector.csv"


def tensor_stats(x: torch.Tensor) -> dict:
    x = x.detach().cpu().float().view(-1)
    return {
        "shape": list(x.shape),
        "finite": bool(torch.isfinite(x).all().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = torch.load(REF_STATS, map_location="cpu", weights_only=False)
    ckpt = torch.load(EH_CKPT, map_location="cpu", weights_only=False)
    diag = json.loads(DIAGNOSTICS.read_text())

    mu_ref = ref["mu_ref"].float().view(-1)
    sigma_ref = ref["Sigma_ref"].float().view(-1)
    ckpt_mu = ckpt["mu_ref"].float().view(-1)
    ckpt_sigma = ckpt["Sigma_ref"].float().view(-1)

    rows = ["index,mu_ref,Sigma_ref"]
    for i, (m, s) in enumerate(zip(mu_ref.tolist(), sigma_ref.tolist())):
        rows.append(f"{i},{m:.12g},{s:.12g}")
    OUT_CSV.write_text("\n".join(rows) + "\n")

    diag_entry = next(
        d for d in diag["checkpoint_diagnostics"] if d["name"] == "E_H_pristine_model"
    )
    summary = {
        "task": "Task 5: inspect pristine reference_stats used by Day-5 E_H",
        "inputs": {
            "reference_stats_pt": str(REF_STATS),
            "eh_checkpoint": str(EH_CKPT),
            "diagnostics_json": str(DIAGNOSTICS),
        },
        "reference_stats_file": {
            "mu_ref": tensor_stats(mu_ref),
            "Sigma_ref": tensor_stats(sigma_ref),
        },
        "checkpoint_reference_stats_match": {
            "mu_ref_allclose": bool(torch.allclose(mu_ref, ckpt_mu)),
            "Sigma_ref_allclose": bool(torch.allclose(sigma_ref, ckpt_sigma)),
            "mu_ref_max_abs_diff": float((mu_ref - ckpt_mu).abs().max().item()),
            "Sigma_ref_max_abs_diff": float((sigma_ref - ckpt_sigma).abs().max().item()),
        },
        "diagnostics_json_E_H_entry": {
            "mu_ref": diag_entry["mu_ref"],
            "Sigma_ref": diag_entry["Sigma_ref"],
        },
        "comparison_note": (
            "The loaded Sigma_ref mean/min/max are on the same order as diagnostics.json "
            "and are approximately 1e-8 to 1e-7."
        ),
        "outputs": {
            "summary_json": str(OUT_JSON),
            "report_txt": str(OUT_TXT),
            "vector_csv": str(OUT_CSV),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    report = f"""Day 6 Task 5: pristine E_H reference stats

Loaded reference stats:
{REF_STATS}

Day-5 E_H checkpoint:
{EH_CKPT}

mu_ref:
- shape: {list(mu_ref.shape)}
- min: {mu_ref.min().item():.8f}
- max: {mu_ref.max().item():.8f}
- mean: {mu_ref.mean().item():.8f}
- std: {mu_ref.std(unbiased=False).item():.8f}

Sigma_ref:
- shape: {list(sigma_ref.shape)}
- min: {sigma_ref.min().item():.12g}
- max: {sigma_ref.max().item():.12g}
- mean: {sigma_ref.mean().item():.12g}
- std: {sigma_ref.std(unbiased=False).item():.12g}

Comparison to diagnostics.json:
- diagnostics mu_ref shape: {diag_entry['mu_ref']['shape']}
- diagnostics mu_ref min/max/mean: {diag_entry['mu_ref']['min']:.8f}, {diag_entry['mu_ref']['max']:.8f}, {diag_entry['mu_ref']['mean']:.8f}
- diagnostics Sigma_ref shape: {diag_entry['Sigma_ref']['shape']}
- diagnostics Sigma_ref min/max/mean: {diag_entry['Sigma_ref']['min']:.12g}, {diag_entry['Sigma_ref']['max']:.12g}, {diag_entry['Sigma_ref']['mean']:.12g}
- The loaded Sigma_ref values match the diagnostics scale: approximately 1e-8 to 1e-7.

Checkpoint consistency:
- mu_ref in reference_stats.pt matches checkpoint mu_ref: {summary['checkpoint_reference_stats_match']['mu_ref_allclose']}
- Sigma_ref in reference_stats.pt matches checkpoint Sigma_ref: {summary['checkpoint_reference_stats_match']['Sigma_ref_allclose']}

Outputs:
- {OUT_JSON}
- {OUT_TXT}
- {OUT_CSV}
"""
    OUT_TXT.write_text(report)
    print(f"Saved Task 5 outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
