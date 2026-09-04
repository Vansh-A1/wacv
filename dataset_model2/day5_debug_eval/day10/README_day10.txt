Day 10 - KADID+TID-only GMM, no KonIQ images

This task fits K=5 diagonal GMM variants using epoch-5 mu vectors only:
- free: sklearn GaussianMixture K=5, diag covariance, kmeans init
- family: family-seeded GaussianMixture, at most K=5 families

Important:
- KonIQ-10k rows are removed before fitting.
- KonIQ-10k images are not loaded or scored.
- Val scoring uses KADID+TID val only.
- Holdout scoring uses KADID holdout only.

Run command:

cd /home/projectwork/student_package
python /home/projectwork/student_package/dataset_model2/day5_debug_eval/day10/day10_fit_score_gmm_no_koniq.py \
  > /home/projectwork/student_package/dataset_model2/day5_debug_eval/day10/day10_run.log 2>&1

Outputs are saved under:
/home/projectwork/student_package/dataset_model2/day5_debug_eval/day10/day10_gmm/ep05/

Expected output files:
- gmm_K05_free.pt
- gmm_K05_family.pt
- fit_log.txt
- family_map.csv
- val_metrics.csv
- holdout_scores.csv
- holdout_metrics.csv
- comparative_table.csv
- day10_report.txt
- Day10_summary.json
- DONE.txt
