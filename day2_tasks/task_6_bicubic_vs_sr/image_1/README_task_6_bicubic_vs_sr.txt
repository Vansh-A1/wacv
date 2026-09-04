Task 6: Bicubic vs SR diagnostic for image_1

Purpose: identify where each SR model output differs from a simple bicubic-upsampled LR baseline.
Important: these difference maps are diagnostic only; they do not prove artifacts exist.

LR input: /home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/DRealSR-20231109T095957Z-004/DRealSR/Test_x4/LR/dreal4_84.png
HR input used for correspondence check: /home/projectwork/Deep_learning/VANSH_WORK/Internship/old/dataset/DRealSR-20231109T095957Z-004/DRealSR/Test_x4/HR/sony_160_x4.png
SwinIR SR input: /data/projectwork/sr_output/swinIr_Inference/SwinIR_results/Dreal/DRealSR_all_x4_sony_160_x1.png
HAT SR input: /data/projectwork/sr_output/HAT_inference/DReal_x4/visualization/DReal_x4/sony_160_x1_DReal_x4.png
Output directory: /home/projectwork/student_package/day2_tasks/task_6_bicubic_vs_sr/image_1

Original LR resolution: 1145x845
Bicubic output resolution: 4580x3380
SR resolution: 4580x3380
Scale factor: 4x
Random seed: 4266
Number of crops per model: 10
Crop size: 512x512
Percentage-pixels-changed threshold: a pixel is changed if any RGB channel has |SR - Bicubic| >= 5 intensity levels.

Subtraction: images were loaded as RGB uint8. For each crop, difference = SR_crop - bicubic_crop using signed integer arrays; abs_difference = abs(difference). Statistics are computed on raw 0-255 RGB absolute differences.
The saved *_absdiff_raw.png files contain raw absolute differences. The *_absdiff_vis_x4.png and comparison panels multiply raw differences by 4 only for visualization.

Coordinates:
crop_01: x=1883, y=2661, width=512, height=512
crop_02: x=2109, y=2849, width=512, height=512
crop_03: x=3958, y=2397, width=512, height=512
crop_04: x=2420, y=1047, width=512, height=512
crop_05: x=2869, y=2078, width=512, height=512
crop_06: x=261, y=419, width=512, height=512
crop_07: x=344, y=1593, width=512, height=512
crop_08: x=3927, y=1207, width=512, height=512
crop_09: x=2244, y=2213, width=512, height=512
crop_10: x=967, y=258, width=512, height=512

Content correspondence check:
The chosen files are paired by the existing DRealSR index/name mapping: dreal4_84 -> sony_160.
swinsr: dreal4_84.png -> sony_160 -> DRealSR_all_x4_sony_160_x1.png; downscaled SR-vs-LR RMSE=4.1177; downscaled HR-vs-LR RMSE=4.4369
hat: dreal4_84.png -> sony_160 -> sony_160_x1_DReal_x4.png; downscaled SR-vs-LR RMSE=5.0794; downscaled HR-vs-LR RMSE=4.4369

Verification:
Original files unchanged: True
Bicubic image exactly 4580x3380: True
SR images exactly 4580x3380: {'swinsr': True, 'hat': True}
Every saved crop/difference image exactly 512x512: True
All generated files are inside image output directory: True

Main outputs:
bicubic/dreal4_84_to_sony_160_bicubic_x4.png
coordinates/crop_coordinates.csv
swinsr/crop_difference_stats.csv
hat/crop_difference_stats.csv
all_crop_difference_stats.csv
reports/verification.json
