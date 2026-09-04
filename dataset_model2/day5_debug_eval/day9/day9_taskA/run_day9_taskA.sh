#!/usr/bin/env bash
set -euo pipefail

cd /home/projectwork/student_package
python /home/projectwork/student_package/dataset_model2/day5_debug_eval/day9/day9_taskA/day9_taskA_fit_gmms.py \
  > /home/projectwork/student_package/dataset_model2/day5_debug_eval/day9/day9_taskA/day9_taskA_run.log 2>&1
