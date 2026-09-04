Reverse diagnostic: SR downsampled to LR vs original LR

Comparison: original LR crop vs SR_downsampled_x4 crop.
Difference name: SR-to-LR reconstruction difference.
This is not an artifact map by itself and is not SR-vs-HR.

Output directory: /home/projectwork/student_package/day2_tasks/reverse_diagnostic_sr_down_vs_lr
Summary CSV: /home/projectwork/student_package/day2_tasks/reverse_diagnostic_sr_down_vs_lr/summary.csv
Source images: 2
Models per image: swinsr, hat
Crop size: 512x512
Base seed: 42; image_1 uses 42, image_2 uses 43.
Downsampling method: cv2.INTER_AREA.
Signed difference: SR_down_crop - LR_crop.
Absolute difference: abs(SR_down_crop - LR_crop).
Percentage thresholds use per-pixel max RGB absolute difference.

Original files unchanged: True
