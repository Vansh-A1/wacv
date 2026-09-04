Task Day 12 / day11b pristine-rank

Output root:
/home/projectwork/student_package/day11/task12/day11b_pristine_rank

This task is the reverse/pristine-rank version:
for severity pair mild x and severe y, train the pristine WACV energy so
E_H(y) is higher than E_H(x) by margin m.

Training pairs:
/home/projectwork/student_package/day11/pairs_severity_train.csv

The pair file is reused from Day11 and must contain only KADID-10k + TID2013
train pairs. It must not contain MOS columns or KonIQ rows.

Initial checkpoint:
/home/projectwork/student_package/checkpoints/hr_combined_ft1/best.pth

Run commands:
bash /home/projectwork/student_package/day11/task12/run_task12_preflight.sh
bash /home/projectwork/student_package/day11/task12/run_task12_train_lambda_0.1.sh
bash /home/projectwork/student_package/day11/task12/run_task12_train_lambda_1.0.sh
bash /home/projectwork/student_package/day11/task12/run_task12_select_holdout.sh

Full run:
bash /home/projectwork/student_package/day11/task12/run_task12_full.sh

Expected outputs:
preflight.json
lambda_0.1/train_log.csv
lambda_0.1/checkpoints/epoch_0005.pth ...
lambda_1.0/train_log.csv
lambda_1.0/checkpoints/epoch_0005.pth ...
val_selection_table.csv
holdout_metrics.csv
compare_vs_unranked_EH.csv
day11b_report.txt
day11b_summary.json
