#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/projectwork/student_package"
DAY_ROOT="$ROOT/day11_day12"
LOG="$DAY_ROOT/auto_start_task12_after_day11.log"

cd "$ROOT"
mkdir -p "$DAY_ROOT/task12/day11b_pristine_rank"

{
  echo "auto_start_task12_after_day11 started at $(date)"
  echo "Waiting for Day11 lambda_1.0 training to stop..."

  while pgrep -f "day11_rankall_pipeline.py train --lambda_rank 1.0" >/dev/null; do
    echo "$(date): Day11 lambda_1.0 still running"
    sleep 60
  done

  echo "$(date): Day11 lambda_1.0 is no longer running"
  if pgrep -f "day11b_pristine_rank_pipeline.py train --lambda_rank 1.0" >/dev/null; then
    echo "$(date): Task12 lambda_1.0 is already running, waiting for it instead of starting a duplicate"
    while pgrep -f "day11b_pristine_rank_pipeline.py train --lambda_rank 1.0" >/dev/null; do
      sleep 60
    done
  else
    echo "$(date): Starting/resuming Task12 lambda_1.0"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" TASK12_EPOCHS="${TASK12_EPOCHS:-25}" \
      bash "$DAY_ROOT/task12/run_task12_train_lambda_1.0.sh"
  fi

  echo "$(date): Task12 lambda_1.0 finished"
  echo "$(date): Running Day11 select+holdout"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" bash "$DAY_ROOT/run_day11_select_holdout.sh"

  echo "$(date): Running Task12 select+holdout"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" bash "$DAY_ROOT/task12/run_task12_select_holdout.sh"

  echo "$(date): all automatic follow-up steps finished"
} 2>&1 | tee -a "$LOG"
