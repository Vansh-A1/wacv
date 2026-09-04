#!/usr/bin/env bash
set -euo pipefail
cd /home/projectwork/student_package
mkdir -p /home/projectwork/student_package/day11_day12/task12/day11b_pristine_rank
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python /home/projectwork/student_package/day11_day12/task12/day11b_pristine_rank_pipeline.py train --lambda_rank 1.0 --epochs "${TASK12_EPOCHS:-25}" \
  2>&1 | tee /home/projectwork/student_package/day11_day12/task12/day11b_pristine_rank/train_lambda_1.0.log
