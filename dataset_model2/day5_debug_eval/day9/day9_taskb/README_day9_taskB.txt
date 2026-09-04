Day 9 Task B - GMM evaluation

This evaluates the 8 Day 9 Task A GMM models:
- ep05: K=2,4,8,16
- ep10: K=2,4,8,16

New computation:
- Encode val KADID+TID images once per checkpoint.
- Encode KADID holdout images once per checkpoint.
- Compute soft GMM energy and hard/min GMM energy for all K.
- Select K per checkpoint using the soft validation K_score.
- Recompute GMM E_A z-normalization using val stats, then build holdout Q_z.

Baseline/ref-only data:
- E_H and ep05 single-Gaussian E_A/Q_z are joined from Day 7 Task D.
- ep10 single-Gaussian E_A/Q_z are joined from Day 8 comparison CSV.
- These baseline scores are not recomputed.

Run command:

cd /home/projectwork/student_package
CUDA_VISIBLE_DEVICES=0 python /home/projectwork/student_package/dataset_model2/day5_debug_eval/day9/day9_taskb/day9_taskB_eval_gmm_models.py \
  > /home/projectwork/student_package/dataset_model2/day5_debug_eval/day9/day9_taskb/taskB_gpu_run.log 2>&1

Outputs:
- val_all8_metrics.csv
- k_selection_ep05.csv
- k_selection_ep10.csv
- holdout_all8_scores.csv
- holdout_metrics.csv
- eval_report.txt
- eval_summary.json
