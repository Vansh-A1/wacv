#!/usr/bin/env bash
set -euo pipefail
cd /home/projectwork/student_package
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python /home/projectwork/student_package/day11_day12/day11_rankall_pipeline.py train --lambda_rank 1.0 --epochs "${DAY11_EPOCHS:-25}" \
  2>&1 | tee /home/projectwork/student_package/day11_day12/day11_train_lambda_1.0.log
