Day 9 Task A - GMM deliverables

This task encodes train-set mu vectors from the Day 7 artifact checkpoints and fits diagonal GMMs for K=2,4,8,16.

Inputs:
- Split manifest:
  /home/projectwork/student_package/dataset_model2/ref_splits_seed42/combined_kadid_tid_koniq_split_seed42.csv
- Epoch 5 checkpoint:
  /home/projectwork/student_package/dataset_model2/checkpoints/wacv_ea_day7_seed42/epoch_0005.pth
- Epoch 10 checkpoint:
  /home/projectwork/student_package/dataset_model2/checkpoints/wacv_ea_day7_seed42/epoch_0010.pth

Output structure:
- day9_gmm/ep05/mu_train.npy
- day9_gmm/ep05/mu_train_ids.csv
- day9_gmm/ep05/gmm_K02.pt, gmm_K04.pt, gmm_K08.pt, gmm_K16.pt
- day9_gmm/ep05/fit_log.txt
- day9_gmm/ep05/DONE.txt
- day9_gmm/ep10/...

Run command:

cd /home/projectwork/student_package
python /home/projectwork/student_package/dataset_model2/day5_debug_eval/day9/day9_taskA/day9_taskA_fit_gmms.py \
  > /home/projectwork/student_package/dataset_model2/day5_debug_eval/day9/day9_taskA/day9_taskA_run.log 2>&1
