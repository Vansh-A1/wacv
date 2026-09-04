Day 8 - quick check for selection criteria

Goal:
1. Recompute selection_score_v2 from the saved Day 7 Task C validation table.
2. Freeze the new selected epoch.
3. Compare epoch 5 (v1) vs v2 selected checkpoint once on KADID holdout only.
4. Do not delete Day 7 Task C outputs or replace them with selection_score_v2.

Main command:

cd /home/projectwork/student_package
python /home/projectwork/student_package/dataset_model2/day8/day8_selection_v2_holdout_compare.py \
  > /home/projectwork/student_package/dataset_model2/day8/day8_run_log.txt 2>&1

Outputs:
- taskC_v2_checkpoint_selection_table.csv
- kadid_holdout_v2_selected_epoch_scores.csv
- kadid_holdout_epoch5_v1_vs_v2_scores.csv
- kadid_holdout_epoch5_v1_vs_v2_metric_compare.csv
- day8_report.txt
- day8_summary.json
