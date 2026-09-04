Day 7 Task D setup

Run from the GPU terminal:

bash /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taske/run_taskD_gpu.sh > /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taske/taskD_gpu_run_log.txt 2>&1

Monitor:

tail -f /home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taske/taskD_gpu_run_log.txt

Selected artifact checkpoint from Task C:
/home/projectwork/student_package/dataset_model2/checkpoints/wacv_ea_day7_seed42/epoch_0005.pth

Frozen pristine E_H checkpoint:
/home/projectwork/student_package/checkpoints/hr_combined_ft1/best.pth

Output CSV:
/home/projectwork/student_package/dataset_model2/day5_debug_eval/day7/day7_taske/taskD_holdout_scores.csv

The script requires CUDA by default. It will stop if GPU is not visible.
