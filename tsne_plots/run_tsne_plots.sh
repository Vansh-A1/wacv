#!/usr/bin/env bash
set -euo pipefail

cd /home/projectwork/student_package
python /home/projectwork/student_package/tsne_plots/make_tsne_plots.py \
  > /home/projectwork/student_package/tsne_plots/tsne_run.log 2>&1
