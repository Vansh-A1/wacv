#!/usr/bin/env bash
set -euo pipefail

cd /home/projectwork/student_package
python /home/projectwork/student_package/dataset_model2/day5_debug_eval/day10/day10_fit_score_gmm_no_koniq.py \
  > /home/projectwork/student_package/dataset_model2/day5_debug_eval/day10/day10_run.log 2>&1
