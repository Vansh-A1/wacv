#!/usr/bin/env bash
set -euo pipefail
cd /home/projectwork/student_package
export MPLCONFIGDIR=/tmp/matplotlib-day7-taskb
python train.py \
  --output_root /home/projectwork/student_package/dataset_model2 \
  --run_name wacv_ea_day7_seed42 \
  --init_from /home/projectwork/student_package/dataset_model2/checkpoints/wacv_degradation_from_scratch_lambda0_seed42/best.pth \
  --init_ref_from /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7_taskb/new_reference_stats_ea_refit.pt \
  --train_list /home/projectwork/student_package/dataset_model2/training_manifests/degradation_train_seed42.txt \
  --eval_list /home/projectwork/student_package/dataset_model2/training_manifests/degradation_val_eval_seed42.txt \
  --epochs 50 \
  --batch_size 8 \
  --threads 4 \
  --seed 42 \
  --img 256 \
  --ldim 100 \
  --lambda_rank 0 \
  --ref_mode once \
  --sz_mode mu_only \
  --sz_sigma_t_max 1.0 \
  --val_every 5 \
  --save_epoch_every 5 \
  --best_metric kadid_srcc \
  --kadid_monitor_csv /home/projectwork/student_package/dataset_model2/training_manifests/kadid_val_monitor_seed42.csv \
  --early_stop_patience 0
