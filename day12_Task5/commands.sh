#!/usr/bin/env bash
set -euo pipefail
cd /home/projectwork/student_package/day12_Task5
export PYTHONDONTWRITEBYTECODE=1
# Run one named stage. Locks and completed stages reject accidental overwrites.
# Original order: prepare, validation, verify validation, kadid, acquire CSIQ,
# extract CSIQ metadata, build CSIQ manifest, csiq, independent verification, report.
case "${1:-verify}" in
  prepare|validation|kadid) python -B run_task5.py "$1" ;;
  csiq) python -B run_task5.py csiq --manifest csiq/evaluation_manifest.csv ;;
  verify)
    python -B verify_outputs.py validation
    python -B verify_outputs.py kadid_holdout
    python -B verify_outputs.py csiq ;;
  report) python -B build_report.py ;;
  *) echo 'Usage: bash commands.sh {prepare|validation|kadid|csiq|verify|report}' >&2; exit 2 ;;
esac
