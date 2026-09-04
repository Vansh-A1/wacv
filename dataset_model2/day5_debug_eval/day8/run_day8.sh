#!/usr/bin/env bash
set -euo pipefail

cd /home/projectwork/student_package
python /home/projectwork/student_package/dataset_model2/day8/day8_selection_v2_holdout_compare.py \
  > /home/projectwork/student_package/dataset_model2/day8/day8_run_log.txt 2>&1
