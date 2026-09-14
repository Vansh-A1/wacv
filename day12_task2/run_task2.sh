#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
python -B -m unittest -v test_task2
python -B run_task2.py --device cuda --batch-size 16 "$@"
