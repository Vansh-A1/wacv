Day 11 severity pairwise rank-all experiment

Output root:
/home/projectwork/student_package/day11

Purpose:
Train E_A using only severity pair ranking on KADID-10k and TID2013 train rows.
No MOS is used for pair mining or training. KonIQ is excluded from pair mining/training
because it has no severity labels. Pristine/reference images are not used as pairs.

Main script:
/home/projectwork/student_package/day11/day11_rankall_pipeline.py

Important inputs:
Train/val/holdout split manifest:
/home/projectwork/student_package/dataset_model2/ref_splits_seed42/combined_kadid_tid_koniq_split_seed42.csv

Initial E_A checkpoint:
/home/projectwork/student_package/dataset_model2/checkpoints/wacv_ea_day7_seed42/epoch_0005.pth

Frozen E_H checkpoint used only for abs_dEH logging in pair CSV:
/home/projectwork/student_package/checkpoints/hr_combined_ft1/best.pth

Commands:
Step A pair generation:
bash /home/projectwork/student_package/day11/run_day11_pairs.sh

Train lambda 0.1:
bash /home/projectwork/student_package/day11/run_day11_train_lambda_0.1.sh

Train lambda 1.0:
bash /home/projectwork/student_package/day11/run_day11_train_lambda_1.0.sh

Selection and holdout evaluation:
bash /home/projectwork/student_package/day11/run_day11_select_holdout.sh

One-shot full run:
bash /home/projectwork/student_package/day11/run_day11_full.sh

Expected core outputs:
pairs_severity_train.csv
pair_stats.txt
day11_rankall/lambda_0.1/checkpoints/epoch_0005.pth ...
day11_rankall/lambda_1.0/checkpoints/epoch_0005.pth ...
val_selection_table.csv
holdout_rankall_scores.csv
holdout_comparison_table.csv
day11_report.txt
