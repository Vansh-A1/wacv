#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON="${TASK_PYTHON:-/opt/conda/bin/python}"
case "${1:-verify}" in
  verify)
    "$TASK_PYTHON" -B "$TASK_DIR/test_distances.py"
    "$TASK_PYTHON" -B "$TASK_DIR/verify_results.py"
    ;;
  *)
    printf '%s\n' 'Usage: bash commands.sh verify' >&2
    exit 2
    ;;
esac
# Execution history (not automatically rerun; reference manifests are frozen):
# 1. prepare_reference_audit.py — copy verified old results, audit and register candidates.
# 2. run_reference_and_preflight.py — first extraction; TF32 replay failure archived.
# 3. Disable TF32 and enable original deterministic settings; repeat extraction/preflight.
# 4. refine_reference_dedup.py — archive superseded pool, exclude RGB duplicates, refreeze.
# 5. run_reference_and_preflight.py — final reference and all 12 pole/arm preflights.
# 6. verify_results.py — independent audit and numerical checks.
# Full matched training is prohibited by the failed PDF preflight. No trainer is launched here.
