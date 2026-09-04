# Graph Report - student_package  (2026-08-29)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 649 nodes · 1401 edges · 40 communities (30 shown, 10 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 49 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 34
- Community 35
- Community 36
- Community 37

## God Nodes (most connected - your core abstractions)
1. `_resize_short_side()` - 40 edges
2. `center_crop()` - 40 edges
3. `main()` - 36 edges
4. `CVAEGenerator_v2` - 28 edges
5. `sz_from_stats()` - 25 edges
6. `score_sz_eval()` - 23 edges
7. `load_best()` - 22 edges
8. `run_validation()` - 15 edges
9. `main()` - 15 edges
10. `main()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `load_best()` --uses--> `CVAEGenerator_v2`  [INFERRED]
  infer.py → external/model.py
- `main()` --uses--> `CVAEGenerator_v2`  [INFERRED]
  train.py → external/model.py
- `main()` --uses--> `EvalImageDataset`  [INFERRED]
  train.py → external/dataloader.py
- `main()` --uses--> `ManifestImageDataset`  [INFERRED]
  train.py → external/dataloader.py
- `run_validation()` --calls--> `_label_from_basename()`  [INFERRED]
  train.py → external/dataloader.py

## Import Cycles
- None detected.

## Communities (40 total, 10 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (59): ema_update_reference(), fit_reference(), load_reference(), Encode pristine images; return mu_ref = mean(mu), Sigma_ref = diag(var(mu))., Exponential moving average of reference stats (on CPU tensors)., Differentiable (posterior-mean) S_Z: uses encoder mu/log_var without…, Kwargs for sz_from_stats / score_sz_* matching the frozen rule. Checkpoint-…, S_Z = sqrt( sum_j (mu_ref_j - mu_t_j)^2 / (Sigma_ref_j + Sigma_t_j) ) where… (+51 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (28): compute_v2_holdout(), finite_corr(), ImagePathDataset, load_model(), main(), make_compare_scores(), metric_rows(), DataFrame (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (51): apply_blur_level_4(), collect_images(), content_proxies(), luminance(), main(), pil_to_float_rgb(), pil_to_tensor(), report_row() (+43 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (22): FastImageDataset, load_model(), main(), Dataset, device, Path, remap_path(), score_image_list() (+14 more)

### Community 4 - "Community 4"
Cohesion: 0.14
Nodes (36): audit_pairing(), bootstrap_ci(), corr_table(), finite_xy(), full_psnr_ssim(), image_mode_size(), main(), partial_rows() (+28 more)

### Community 5 - "Community 5"
Cohesion: 0.17
Nodes (22): add_gmm_columns(), build_holdout_scores(), corr(), encode_mu(), gmm_energies(), gmm_val_stats(), holdout_metrics(), ImageDataset (+14 more)

### Community 6 - "Community 6"
Cohesion: 0.22
Nodes (21): checkpoint_diagnostics(), compute_correlations(), corr_pair(), diagnose_reference_set(), distortion_bar(), load_holdout_manifest(), load_image_tensor(), load_model() (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.24
Nodes (21): add_arm_scores(), build_family_map(), build_holdout_scores(), comparative_table(), corr(), family_for_distortion(), fit_family(), fit_free() (+13 more)

### Community 8 - "Community 8"
Cohesion: 0.20
Nodes (18): checkpoint_label(), choose_checkpoint(), corr(), distortion_group(), ImagePathDataset, load_model(), main(), make_plots() (+10 more)

### Community 9 - "Community 9"
Cohesion: 0.19
Nodes (17): build_metrics(), corr_row(), ImagePathDataset, load_model(), load_or_recompute_scores(), main(), make_plot(), metric() (+9 more)

### Community 10 - "Community 10"
Cohesion: 0.18
Nodes (17): compute_correlations(), epoch_from_path(), load_model(), load_validation_rows(), main(), make_plots(), DataFrame, Dataset (+9 more)

### Community 11 - "Community 11"
Cohesion: 0.19
Nodes (16): apply_stats(), ImagePathDataset, load_koniq_mos(), load_manifest(), load_model(), main(), DataFrame, Dataset (+8 more)

### Community 12 - "Community 12"
Cohesion: 0.26
Nodes (19): build_lr_map(), center_metric_crop(), collect_images(), find_zssr_folders(), iteration_number(), lr_content(), lr_number(), main() (+11 more)

### Community 13 - "Community 13"
Cohesion: 0.19
Nodes (15): compare_vectors(), corr_row(), cosine(), encode_paths(), ImagePathDataset, load_ea_model(), load_fixed_200_diagnostic_rows(), main() (+7 more)

### Community 14 - "Community 14"
Cohesion: 0.22
Nodes (15): encode_mu(), fit_and_save_gmms(), ImageDataset, load_encoder(), load_train_manifest(), main(), DataFrame, Dataset (+7 more)

### Community 15 - "Community 15"
Cohesion: 0.25
Nodes (17): center_metric_crop(), finite_pairs(), load_existing(), main(), metric_correlation_rows(), pil_rgb(), pil_to_tensor(), psnr_rgb() (+9 more)

### Community 16 - "Community 16"
Cohesion: 0.29
Nodes (15): comparison(), crop_coord(), downsample_area(), ImageCase, main(), make_fresh_output_dir(), pil_rgb(), Image (+7 more)

### Community 17 - "Community 17"
Cohesion: 0.19
Nodes (11): corr_row(), encode_mu(), ImagePathDataset, load_eh_model(), main(), make_plot(), DataFrame, Dataset (+3 more)

### Community 18 - "Community 18"
Cohesion: 0.32
Nodes (14): corr_values(), e1_mos_tables(), e2_overall_severity(), e3_within_type_severity(), e4_day6_vs_day7(), finite_xy(), main(), DataFrame (+6 more)

### Community 19 - "Community 19"
Cohesion: 0.31
Nodes (14): csv_write(), ensure_rgb(), FileState, image_rmse(), main(), make_coords(), move_previous_root_outputs(), Image (+6 more)

### Community 20 - "Community 20"
Cohesion: 0.25
Nodes (10): analyze_ea_ref_paths(), compute_metrics(), main(), plot_vector_comparisons(), DataFrame, Path, Tensor, Generate visual comparison plots for mu_ref and Sigma_ref. (+2 more)

### Community 21 - "Community 21"
Cohesion: 0.31
Nodes (9): compute_correlations_koniq(), corr_pair(), main(), plot_scatter_koniq(), DataFrame, Path, 3 scatter plots as specified: 1. E_H vs KonIQ MOS -> KonIQ_E_H_vs_mos.png 2.…, Compute Spearman, Pearson, Kendall for a pair of arrays. (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.42
Nodes (9): build_kadid(), build_tid2013(), main(), DataFrame, Path, split_ref_ids(), tid_ref_map(), verify_reference_exclusivity() (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.36
Nodes (7): main(), DataFrame, Path, Series, safe_copy(), select_cases(), zscore()

### Community 24 - "Community 24"
Cohesion: 0.57
Nodes (6): assign_splits(), main(), make_combined_manifest(), make_manifest(), DataFrame, Path

### Community 25 - "Community 25"
Cohesion: 0.70
Nodes (4): correlation_rows(), load_and_adjust(), main(), plot_metric()

### Community 26 - "Community 26"
Cohesion: 0.67
Nodes (3): main(), Tensor, tensor_stats()

## Knowledge Gaps
- **10 isolated node(s):** `ImageCase`, `MPLCONFIGDIR`, `train_command.sh script`, `MPLCONFIGDIR`, `run_taskD_gpu.sh script` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CVAEGenerator_v2` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 5`, `Community 6`, `Community 8`, `Community 9`, `Community 10`, `Community 11`, `Community 13`, `Community 14`, `Community 17`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `center_crop()` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 8`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 17`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `_resize_short_side()` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 8`, `Community 9`, `Community 10`, `Community 11`, `Community 12`, `Community 13`, `Community 14`, `Community 15`, `Community 17`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `_resize_short_side()` (e.g. with `main()` and `center_metric_crop()`) actually correct?**
  _`_resize_short_side()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `center_crop()` (e.g. with `main()` and `center_metric_crop()`) actually correct?**
  _`center_crop()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `main()` (e.g. with `EvalImageDataset` and `ManifestImageDataset`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `CVAEGenerator_v2` (e.g. with `load_best()` and `main()`) actually correct?**
  _`CVAEGenerator_v2` has 2 INFERRED edges - model-reasoned connections that need verification._