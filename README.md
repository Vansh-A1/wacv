# CVAE Image Quality Score (student package)

This folder is a small, ready-to-use package for students.

You can:
1. Score images with the trained model (inference)
2. Continue training / fine-tune on your own clean images

---

## What this model does

The model looks at one image and outputs a number called **S_Z**.

- **Higher S_Z = worse quality**
- **Lower S_Z = better quality**

It does **not** need a reference (clean) image at test time.

---

## Folder contents

```
student_package/
  infer.py                 # score images
  train.py                 # train / fine-tune
  score.py                 # scoring helpers (used by the scripts above)
  requirements.txt
  configs/scoring_frozen.yaml
  checkpoints/best.pth     # trained model (v1.5a)
  external/                # model + dataloader code
  examples/
    sample_images/         # a few demo images
    train_list_example.txt
    eval_list_example.txt
```

---

## Setup

```bash
cd student_package
pip install -r requirements.txt
```

You need Python 3 and PyTorch. A GPU is strongly recommended for training.
Inference can run on CPU (slower).

---

## 1) Inference (score images)

Score one image:

```bash
python infer.py --image examples/sample_images/002963_HR.png
```

Score a whole folder:

```bash
python infer.py --folder examples/sample_images --out_dir test_results
```

Useful options:
- `--ckpt checkpoints/best.pth`  (default)
- `--cpu`  force CPU
- `--save_recon`  also save input|reconstruction image pairs
- `--gpu 0`  choose GPU id

Results are printed and also saved to `test_results/scores.csv`.

Expected demo behavior: the clean `*_HR.png` images should get a **lower** S_Z than the blur/jpeg images.

---

## 2) Fine-tuning (continue training)

### Step A — make a train list

Create a text file with **one clean image path per line**.

Example format (`examples/train_list_example.txt`):

```
/path/to/my_clean_images/img001.png
/path/to/my_clean_images/img002.png
```

Use only pristine / high-quality images for training.

### Step B — make an eval list (optional but recommended)

Use a mix of clean and degraded images.
You can start from `examples/eval_list_example.txt`.

### Step C — run fine-tuning

```bash
python train.py \
  --train_list /path/to/my_train_list.txt \
  --eval_list examples/eval_list_example.txt \
  --init_from checkpoints/best.pth \
  --ver ft1 \
  --epochs 50 \
  --batch_size 8 \
  --gpu 0 \
  --amp
```

Notes:
- By default, training **starts from** `checkpoints/best.pth` (warm start).
- By default, the frozen quality reference (`mu_ref` / `Sigma_ref`) inside that checkpoint is **kept** (not refit).
- New runs are saved under:
  - `checkpoints/..._vft1/best.pth`
  - `runs/..._vft1/`
- If a run name already exists, bump `--ver` (example: `--ver ft2`) or add `--resume`.

After fine-tuning, score with your new checkpoint:

```bash
python infer.py --ckpt checkpoints/<your_run_name>/best.pth --folder examples/sample_images
```

---

## Important rules (please read)

1. **Higher S_Z = worse.** Do not flip this.
2. Prefer keeping the reference stats from `best.pth` unless your experiment is specifically about refitting them.
3. Training needs a GPU. Inference can use `--cpu`.
4. Images are cropped to **256x256**. Very small images are resized up first.

---

## Quick check that the package works

```bash
python infer.py --folder examples/sample_images --out_dir test_results
cat test_results/scores.csv
```

If this prints S_Z numbers and creates `scores.csv`, you are ready to go.
