#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_PYTHON="${TASK_PYTHON:-/opt/conda/bin/python}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
case "${1:-status}" in
 status) cat "$TASK_DIR/status.json"; if [[ -f "$TASK_DIR/progress.json" ]]; then cat "$TASK_DIR/progress.json"; fi ;;
 verify) "$TASK_PYTHON" -B -c "import sys; sys.path.insert(0, sys.argv[1]); from common import verify_frozen; verify_frozen(); print('Frozen amendment and inputs verified')" "$TASK_DIR" ;;
 run) "$TASK_PYTHON" -B "$TASK_DIR/run_pipeline.py" --phase all ;;
 *) printf '%s\n' 'Usage: bash commands.sh status|verify|run' >&2; exit 2 ;;
esac
