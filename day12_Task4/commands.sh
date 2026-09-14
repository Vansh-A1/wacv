#!/usr/bin/env bash
set -euo pipefail
cd /home/projectwork/student_package/day12_Task4
# Prerequisite audit only. Exit code 2 means BLOCKED, not successful training.
# This command never opens images, creates model checkpoints, or runs an optimizer.
PYTHONDONTWRITEBYTECODE=1 /opt/conda/bin/python -B audit_prerequisites.py --config config.json
