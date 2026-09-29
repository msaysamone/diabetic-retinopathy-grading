# Diabetic retinopathy grading: EyePACS → APTOS

A ResNet50 that grades diabetic retinopathy (DR) from fundus photographs on the 5-point ICDR scale (0 none, 1 mild, 2 moderate, 3 severe, 4 proliferative). It is pretrained on EyePACS (stage 1), then fine-tuned on APTOS 2019 (stage 2). Everything was trained on a single 8 GB RTX 4060.

## Results

Final evaluation on the **held-out APTOS test split** (366 images, never used for training or model selection). The model, metrics and referral rule were fixed before the test set was touched, and the test set was run once.

| Model | QWK | Referable-DR AUC | Referral sensitivity | Referral specificity | Recall by grade 0 / 1 / 2 / 3 / 4 |
|---|---|---|---|---|---|
| Dev-selected | 0.908 (0.879–0.934) | 0.988 (0.980–0.995) | 94.9% (90.8–98.4) | 93.4% (90.0–96.5) | .98 / .47 / .83 / .41 / .45 |
| Refit (train + dev) | 0.912 (0.884–0.938) | 0.988 (0.980–0.995) | 95.6% (91.9–98.6) | 93.4% (90.0–96.5) | .98 / .47 / .84 / .47 / .52 |

Brackets are bootstrap 95% confidence intervals. *Referable* means grade ≥ 2, and a case is referred when the predicted grade is ≥ 2. Full confusion matrices are in [`FINAL_EVAL.md`](FINAL_EVAL.md).

- **Referral holds up.** Referral sensitivity and specificity (about 95% / 93%) match the dev split almost exactly, and AUC is slightly higher on test.
- **Grading is a bit weaker than on dev.** Test QWK is 0.908–0.912, versus 0.923 on dev. That's within dev's confidence interval and expected, since dev was used to pick the epoch.
- **Refit vs dev-selected makes no real difference.** QWK is +0.004 (paired 95% CI −0.005 to +0.015), and both models make identical decisions on every non-referable image.
- **Weak spots are the rare and borderline grades:**
  - Proliferative DR (grade 4) is often under-graded: 10 of 33 are called moderate. They're still referred, but at the wrong urgency.
  - One proliferative case is graded mild by both models, so it isn't referred.
  - About half of the mild cases (14 of 30) are graded moderate. This is the main source of false referrals.

### APTOS 2019 competition (late submission)

The same two models were also submitted after the deadline to the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) competition. It's a code competition: Kaggle runs the inference notebook on its hidden test set of about 13,000 images. These images are not part of the APTOS split used above; that split was carved from the competition's training set.

| Model | Public QWK | Private QWK |
|---|---|---|
| Dev-selected | 0.779 | 0.899 |
| Refit (train + dev) | 0.775 | 0.900 |

- **Private QWK of 0.90** is close to the 0.908–0.912 on our own test split. That's despite the hidden set's different framing (lower-resolution images, with the fundus often cut off at the top and bottom) and very different grade mix.
- **The public score is much lower,** as it was for most teams in this competition. The public set is a small slice (about 1,900 images) with an unusual grade mix. The leaderboard was ranked on the private score.
- **Winning solutions scored about 0.93–0.94 private,** using ensembles of several larger models, larger inputs and pseudo-labelling. This is a single ResNet50 at 512 px with no test-time augmentation.
- Late submissions aren't ranked on the leaderboard.

## What was tried

Each iteration changes one thing from the baseline. All results are on the **APTOS dev split** (366 images) at the best epoch. The Δ interval is a paired bootstrap against the baseline: both runs are scored on the same resampled images. A change counts as an improvement only if that interval is above 0.

| Iteration | Change | Dev QWK | Δ vs baseline (paired 95% CI) | Referable AUC |
|---|---|---|---|---|
| `iter1` | **Baseline:** weighted cross-entropy, hard labels, augmentation | **0.923** | — | 0.976 |
| `iter1_seed43` | Baseline with another seed (noise check) | 0.923 | +0.000 (−0.005 to +0.008) | 0.976 |
| `iter2_soft` | Ordinal soft labels (ε = 0.2 spread onto neighbouring grades) | 0.912 | −0.011 (−0.023 to 0.000) | 0.973 |
| `iter3_mixup` | Mixup (α = 0.2) | 0.913 | −0.010 (−0.025 to +0.004) | 0.976 |
| `iter4_kappa` | Differentiable QWK loss added to cross-entropy | 0.921 | −0.002 (−0.011 to +0.007) | 0.976 |
| `iter5_regression` | Regression: one output, weighted MSE, rounded to the nearest grade | 0.916 | −0.007 (−0.026 to +0.014) | 0.969 |
| `iter6_sched20` | Cosine LR schedule that reaches 0 at epoch 20, no early stopping | 0.918 | −0.004 (−0.012 to +0.002) | 0.976 |

Iterations 2–6 fine-tune from the same stage-1 checkpoint, so only stage 2 differs.

**None of the changes beat plain weighted cross-entropy.**
- **Soft labels and mixup** are about 0.01 worse, a small but consistent loss. The noise check shows run-to-run variation of about ±0.005.
- **Kappa loss and the longer schedule** tie with the baseline.
- **Regression squeezes its predictions toward the middle of the scale.** Grade 3 recall rises but grade 4 recall drops, from 0.71 to 0.39, because uncertain proliferative cases get averaged down toward 3.

**Decoding was also tested, with no gain** ([`DECODING.md`](DECODING.md)). Instead of taking the most likely grade, the alternative uses the expected grade Σ g·p(g) with cut-points tuned to maximise QWK:
- **Tuned on all of dev**, the cut-points look like a gain (+0.004 to +0.008).
- **Cross-validated within dev**, they are 0.003–0.013 *worse* for every run.

The apparent gain is overfitting 4 cut-points to 366 images.

Per-run details (commands, settings, per-stage scores) are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Method

### Data

| Dataset | Use | Images |
|---|---|---|
| [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection) (Kaggle 2015, train set) | Stage 1 training | 33,370 train / 1,756 holdout |
| [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection), split from [`mariaherrerot/aptos2019`](https://www.kaggle.com/datasets/mariaherrerot/aptos2019) | Stage 2 training / model selection / final test | 2,930 train / 366 dev / 366 test |

- **EyePACS holdout** (`split.ipynb`) is 5% of patients. The split is patient-level, so both eyes of a patient land on the same side. It's stratified by each patient's worse-eye grade, with seed 42.
- **APTOS** keeps the published train / valid / test split. Here "valid" is called *dev*.

### Preprocessing (`preprocess.py`)

Each image is cropped to the fundus (the dark border is removed), padded to a black square and resized to 512 × 512 PNG. There is no colour normalisation (such as Ben Graham's method or CLAHE).

### Training (`train.py`, same code as `train.ipynb`)

- **Model:** ResNet50 with ImageNet weights, fully fine-tuned, at 512 px, batch 24. It uses bf16 autocast and `torch.compile`.
- **Loss:** cross-entropy, with class weights of 1/√(class count) and natural class balance (no resampling).
- **Augmentation:** horizontal and vertical flips, 0–360° rotation, a small random crop (90–100% scale), and ±20% brightness and contrast.
- **Optimiser:** AdamW with learning rate 1e-4 for the backbone and 1e-3 for the head, weight decay 1e-4. Cosine decay per step over 30 epochs.
- **Model selection:** early stopping after 5 epochs without improvement in validation QWK. The epoch with the best validation QWK is kept, so QWK is the selection metric, not the loss.
- **Stage 1:** trained on EyePACS and validated on the EyePACS holdout. Best holdout QWK was 0.796 (AUC 0.954), at epoch 8.
- **Stage 2:** fine-tunes the stage-1 checkpoint on APTOS train with a fresh optimiser, learning rates ÷ 10 and APTOS class weights, validated on APTOS dev. Best epoch was 13.
- **Refit:** retrains stage 2 on APTOS train + dev for the same 13 epochs. There's no validation set left, so the epoch count comes from stage 2.

Each run is configured by environment variables, which are recorded in `run_config.json`:

| Variable | Default | Meaning |
|---|---|---|
| `RUN`, `DESC` | `iter1` | Run name (checkpoints go to `data/checkpoints/<RUN>/`) and a one-line description |
| `SOFT_EPS` | `0` | Ordinal soft labels |
| `MIXUP_ALPHA` | `0` | Mixup |
| `KAPPA_WEIGHT` | `0` | Weight of the soft-QWK loss term |
| `REGRESSION` | `0` | `1` = one-output regression with weighted MSE |
| `MAX_EPOCHS`, `PATIENCE` | `30`, `5` | Epoch limit (also the cosine schedule length) and early-stopping patience |
| `STAGE1_CKPT` | | Reuse a stage-1 checkpoint (stage-2-only runs) |
| `REFIT` | `1` | Retrain on train + dev after stage 2 |
| `SEED` | `42` | Random seed |

### Evaluation protocol

- **APTOS dev** is used for everything that involves a choice: early stopping, comparing iterations, and decoding.
- **APTOS test** was used once, at the end, by `evaluate.py --test`. Before that run, the following were fixed:
  - the model: iter1, reporting both the dev-selected and refit versions;
  - the metrics;
  - the referral rule: predicted grade ≥ 2.
- **Intervals** are percentile bootstraps (2,000 resamples of images). Differences between models use paired resamples.

## Reproducing

```bash
python preprocess.py                 # download (Kaggle API) and preprocess EyePACS + APTOS into data/processed/
jupyter nbconvert --execute split.ipynb   # EyePACS patient-level holdout + APTOS split CSVs into splits/
python train.py                      # iter1: stage 1, stage 2, refit
# stage-2-only variants, e.g.:
RUN=iter4_kappa KAPPA_WEIGHT=1 STAGE1_CKPT=data/checkpoints/iter1/stage1_eyepacs/best.pt REFIT=0 python train.py
python summarize.py                  # -> EXPERIMENTS.md
python decode.py                     # -> DECODING.md
python evaluate.py                   # dry run on dev; `--test` for the final test run -> FINAL_EVAL.md
```

`data/` is a symlink to external storage and isn't included: about 10 GB of processed images and 4 GB of checkpoints.

**Rough timings on an RTX 4060:**
- Stage 1: about 7.6 min per epoch (13 epochs).
- Stage 2: about 0.7 min per epoch.

## Limitations

- **Small evaluation sets.** APTOS dev and test have only 17–40 images each in grades 1, 3 and 4, so per-grade recall moves by 3–6 points per image, and confidence intervals are wide.
- **One stage-1 checkpoint.** All stage-2 iterations reuse iter1's stage 1, which was picked at epoch 8 with the learning rate still about 83% of its peak. Changing stage 1 (for example, applying soft labels or mixup there too) wasn't tested.
- **The learning rate was never tuned.** Every iteration uses the baseline's learning rate, so "no improvement" means no improvement at these settings.
- **Limited external validation.** The competition's hidden test set shows the model holds up under a shift in framing and grade mix, but it's still APTOS data (the same Aravind Eye Hospital program). Messidor-2, the planned out-of-domain test with adjudicated grades from Krause et al. 2018, hasn't been evaluated. So the README doesn't yet show how the model handles other cameras or populations.
- **Not a medical device.** This is a research project. It isn't validated for clinical use.
