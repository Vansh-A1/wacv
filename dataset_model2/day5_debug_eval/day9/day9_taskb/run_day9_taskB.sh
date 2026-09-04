#!/usr/bin/env bash
set -euo pipefail

cd /home/projectwork/student_package
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python /home/projectwork/student_package/dataset_model2/day5_debug_eval/day9/day9_taskb/day9_taskB_eval_gmm_models.py \
  > /home/projectwork/student_package/dataset_model2/day5_debug_eval/day9/day9_taskb/taskB_gpu_run.log 2>&1
