Task 6: Bicubic vs SR diagnostic for image_2

Purpose: identify where each SR model output differs from a simple bicubic-upsampled LR baseline.
Important: these difference maps are diagnostic only; they do not prove artifacts exist.

LR input: /home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/DRealSR-20231109T095957Z-004/DRealSR/Test_x4/LR/dreal4_1.png
HR input used for correspondence check: /home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/DRealSR-20231109T095957Z-004/DRealSR/Test_x4/HR/Canon_10_x4.png
SwinIR SR input: /data/projectwork/sr_output/swinIr_Inference/SwinIR_results/Dreal/DRealSR_all_x4_Canon_10_x1.png
HAT SR input: /data/projectwork/sr_output/HAT_inference/DReal_x4/visualization/DReal_x4/Canon_10_x1_DReal_x4.png
Output directory: /home/projectwork/student_package/day2_tasks/task_6_bicubic_vs_sr/image_2

Original LR resolution: 1445x945
Bicubic output resolution: 5780x3780
SR resolution: 5780x3780
Scale factor: 4x
Random seed: 4267
Number of crops per model: 10
Crop size: 512x512
Percentage-pixels-changed threshold: a pixel is changed if any RGB channel has |SR - Bicubic| >= 5 intensity levels.

Subtraction: images were loaded as RGB uint8. For each crop, difference = SR_crop - bicubic_crop using signed integer arrays; abs_difference = abs(difference). Statistics are computed on raw 0-255 RGB absolute differences.
The saved *_absdiff_raw.png files contain raw absolute differences. The *_absdiff_vis_x4.png and comparison panels multiply raw differences by 4 only for visualization.

Coordinates:
crop_01: x=2677, y=319, width=512, height=512
crop_02: x=3931, y=2330, width=512, height=512
crop_03: x=5131, y=1390, width=512, height=512
crop_04: x=1537, y=2549, width=512, height=512
crop_05: x=859, y=1206, width=512, height=512
crop_06: x=2907, y=3119, width=512, height=512
crop_07: x=4894, y=1808, width=512, height=512
crop_08: x=466, y=2266, width=512, height=512
crop_09: x=3454, y=1776, width=512, height=512
crop_10: x=1033, y=351, width=512, height=512

Content correspondence check:
The chosen files are paired by the existing DRealSR index/name mapping: dreal4_1 -> Canon_10.
swinsr: dreal4_1.png -> Canon_10 -> DRealSR_all_x4_Canon_10_x1.png; downscaled SR-vs-LR RMSE=13.4115; downscaled HR-vs-LR RMSE=6.1109
hat: dreal4_1.png -> Canon_10 -> Canon_10_x1_DReal_x4.png; downscaled SR-vs-LR RMSE=11.9665; downscaled HR-vs-LR RMSE=6.1109

Verification:
Original files unchanged: True
Bicubic image exactly 4580x3380: True
SR images exactly 4580x3380: {'swinsr': True, 'hat': True}
Every saved crop/difference image exactly 512x512: True
All generated files are inside image output directory: True

Main outputs:
bicubic/dreal4_1_to_Canon_10_bicubic_x4.png
coordinates/crop_coordinates.csv
swinsr/crop_difference_stats.csv
hat/crop_difference_stats.csv
all_crop_difference_stats.csv
reports/verification.json
