#!/usr/bin/env bash
set -euo pipefail
cd /home/projectwork/student_package
bash /home/projectwork/student_package/day11_day12/run_day11_pairs.sh
bash /home/projectwork/student_package/day11_day12/run_day11_train_lambda_0.1.sh
bash /home/projectwork/student_package/day11_day12/run_day11_train_lambda_1.0.sh
bash /home/projectwork/student_package/day11_day12/run_day11_select_holdout.sh
