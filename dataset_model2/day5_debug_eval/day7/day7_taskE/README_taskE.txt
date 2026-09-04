Day 7 Task E - Spearman tables

This folder contains the Task E script and outputs.

The script uses:
- Day 7 Task D holdout/val scores:
  /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taskD/
- Day 7 Task C selected E_A checkpoint table:
  /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taskc/taskc_checkpoint_selection_table.csv
- Day 7 new artifact reference comparison:
  /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taskA/day7_reference_stats_comparison.csv
- Day 6 baseline outputs:
  /home/projectwork/student_package/dataset_model2/day5_debug_eval/day6_task/

Run command:

cd /home/projectwork/student_package
python /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taskE/day7_taskE_spearman_tables.py \
  > /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taskE/taskE_run_log.txt 2>&1

Note: this task computes correlations from existing CSV scores, so it does not require GPU.
