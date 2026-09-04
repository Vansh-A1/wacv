#!/usr/bin/env bash
set -euo pipefail
cd /home/projectwork/student_package
python /home/projectwork/student_package/day11_day12/make_day11_day12_tsne.py --cpu \
  2>&1 | tee /home/projectwork/student_package/day11_day12/day11_day12_tsne_run.log
