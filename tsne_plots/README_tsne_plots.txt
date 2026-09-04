t-SNE plot task completed

Four plots were made on the same validation set with identical t-SNE settings.
Validation set: /home/projectwork/student_package/dataset_model2/day5_debug_eval/day9/day9_taskb/encoded_mu_cache/ep05_val_kadid_tid_ids.csv
N: 1860
Dataset counts: {'KADID-10k': 1500, 'TID2013': 360}
Group B counts: {'other': 1155, 'noise': 420, 'blur': 210, 'HF/sharpen': 75}

t-SNE settings:
{
  "seed": 42,
  "perplexity": 30.0,
  "max_iter": 1000,
  "init": "pca",
  "learning_rate": "auto",
  "metric": "euclidean"
}

Outputs:
/home/projectwork/student_package/tsne_plots/01_EA_epoch5_A_by_distortion_type.png
/home/projectwork/student_package/tsne_plots/02_EA_epoch5_B_by_group.png
/home/projectwork/student_package/tsne_plots/03_EH_frozen_A_by_distortion_type.png
/home/projectwork/student_package/tsne_plots/04_EH_frozen_B_by_group.png
/home/projectwork/student_package/tsne_plots/tsne_coordinates_same_valset.csv
