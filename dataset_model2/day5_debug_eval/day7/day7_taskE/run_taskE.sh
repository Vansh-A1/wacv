#!/usr/bin/env bash
set -euo pipefail

cd /home/projectwork/student_package
python /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taskE/day7_taskE_spearman_tables.py \
  > /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taskE/taskE_run_log.txt 2>&1
