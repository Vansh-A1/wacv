#!/usr/bin/env bash
set -euo pipefail
cd /home/projectwork/student_package
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python /home/projectwork/student_package/day11_day12/day11_rankall_pipeline.py select \
  2>&1 | tee /home/projectwork/student_package/day11_day12/day11_select.log
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python /home/projectwork/student_package/day11_day12/day11_rankall_pipeline.py holdout \
  2>&1 | tee /home/projectwork/student_package/day11_day12/day11_holdout.log
