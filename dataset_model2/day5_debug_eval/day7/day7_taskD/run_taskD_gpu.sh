#!/usr/bin/env bash
set -euo pipefail
cd /home/projectwork/student_package
export MPLCONFIGDIR=/tmp/matplotlib-day7-taske
python /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taske/day7_taskd_holdout_qz_scores.py \
  --batch_size 64 \
  --workers 4
