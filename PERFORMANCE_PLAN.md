# deepBlink Performance Improvement Plan

Systematic evaluation of training quality and inference speed improvements,
benchmarked on a SLURM cluster against one reference dataset.

---

## 1. Overview

Each improvement is implemented as a **standalone config flag or code change**
on the branch `claude/explore-performance-improvements-j7WRO`.
A single SLURM array job trains every variant and evaluates it on the test
split, producing a CSV summary table at the end.

### Metrics collected per experiment

| Metric | Source |
|---|---|
| F1 @ 3 px | `metrics.compute_metrics(mdist=3)` |
| F1 integral @ 3 px | area under F1 vs. cutoff |
| Mean euclidean distance | average offset of matched spots |
| Training wall-clock time | SLURM `sacct -j` |
| Inference time per image | `time.perf_counter` around `model.predict` |
| Peak GPU memory | `nvidia-smi --query-gpu=memory.used` |

---

## 2. Experiment matrix

Each row is one SLURM array task. The **baseline** row uses the current
defaults so every other row changes exactly one thing.

| ID | Name | What changes | Files touched |
|----|------|-------------|---------------|
| 0 | `baseline` | Current defaults (no augmentation, lr=1e-4 flat, bs=2, combined_dice_rmse, amsgrad) | config only |
| 1 | `aug_all` | Enable all 5 augmentations | config only |
| 2 | `aug_flip_rotate` | Enable flip + rotate only | config only |
| 3 | `lr_cosine` | Cosine-annealing LR schedule (1e-3 -> 1e-6) | `training.py` callback |
| 4 | `lr_plateau` | ReduceLROnPlateau (patience=10, factor=0.5) | `training.py` callback |
| 5 | `lr_warmup_cosine` | Linear warmup 10 epochs + cosine decay | `training.py` callback |
| 6 | `bs8` | Batch size 8, lr scaled 4x (4e-4) | config only |
| 7 | `bs16` | Batch size 16, lr scaled 8x (8e-4) | config only |
| 8 | `early_stop` | EarlyStopping patience=20, restore_best_weights | `training.py` callback |
| 9 | `focal_loss` | Focal loss (gamma=2) for probability + RMSE for coords | `losses.py` new fn |
| 10 | `smooth_09` | Label smoothing factor = 0.9 | config only |
| 11 | `combined_best` | aug_all + warmup_cosine + bs8 + early_stop + focal + smooth_09 | all above |
| 12 | `mixed_precision` | `keras.mixed_precision.set_global_policy("mixed_float16")` | `training.py` top |
| 13 | `tta` | Test-time augmentation (4x rot + 2x flip, average) | `inference.py` |
| 14 | `combined_best_tta` | Experiment 11 model + TTA at inference | reuse model 11 |

> Experiments 0-11 measure **training quality**. Experiments 12-14 additionally
> measure **speed** effects.

---

## 3. Implementation plan

### 3.1 Config generation script

Create `experiments/generate_configs.py` that:

1. Reads the default config template from `deepblink.cli._config.HandleConfig`.
2. For each experiment ID, patches the relevant keys.
3. Writes `experiments/configs/exp_{ID}_{name}.yaml`.

This keeps every experiment fully reproducible from its YAML alone.

### 3.2 Code changes (minimal, flag-gated)

All changes live in a small number of files.  Nothing breaks existing
behaviour; new code paths are only activated by config keys.

#### `deepblink/losses.py` -- add focal loss

```python
def focal_loss(gamma=2.0):
    """Focal loss factory for the probability channel."""
    def _focal(y_true, y_pred):
        bce = keras.losses.binary_crossentropy(
            ops.flatten(y_true), ops.flatten(y_pred)
        )
        p_t = ops.exp(-bce)
        return ops.mean((1 - p_t) ** gamma * bce)
    _focal.__name__ = "focal_loss"
    return _focal

def combined_focal_rmse(y_true, y_pred):
    """Focal loss for probability + RMSE for coordinates."""
    return focal_loss()(y_true[..., 0], y_pred[..., 0]) + rmse(y_true, y_pred) * 2
```

#### `deepblink/training.py` -- LR schedule & early stopping callbacks

Add a helper that reads optional keys from `cfg["train_args"]`:

```python
def _build_callbacks(cfg, base_callbacks):
    ta = cfg.get("train_args", {})

    if ta.get("early_stopping", False):
        base_callbacks.append(keras.callbacks.EarlyStopping(
            patience=ta.get("early_stopping_patience", 20),
            restore_best_weights=True,
        ))

    schedule = ta.get("lr_schedule", None)
    if schedule == "cosine":
        base_callbacks.append(keras.callbacks.LearningRateScheduler(
            lambda epoch: ...  # cosine formula
        ))
    elif schedule == "plateau":
        base_callbacks.append(keras.callbacks.ReduceLROnPlateau(
            patience=10, factor=0.5
        ))
    elif schedule == "warmup_cosine":
        # linear warmup for warmup_epochs, then cosine
        ...

    if ta.get("mixed_precision", False):
        keras.mixed_precision.set_global_policy("mixed_float16")

    return base_callbacks
```

#### `deepblink/inference.py` -- TTA wrapper

```python
def predict_tta(image, model, probability=0.5):
    """Predict with test-time augmentation (rotations + flips)."""
    preds = []
    for k in range(4):                       # 0/90/180/270
        img_rot = np.rot90(image, k)
        coords = predict(img_rot, model, probability)
        coords = _undo_rotation(coords, k, image.shape)
        preds.append(coords)
    for axis in [0, 1]:                      # horizontal / vertical flip
        img_flip = np.flip(image, axis)
        coords = predict(img_flip, model, probability)
        coords = _undo_flip(coords, axis, image.shape)
        preds.append(coords)
    return _merge_predictions(preds, merge_radius=2.0)
```

### 3.3 Evaluation script

Create `experiments/evaluate.py`:

```
Usage: python experiments/evaluate.py \
    --model  experiments/models/exp_0_baseline.h5 \
    --dataset /path/to/dataset.npz \
    --output experiments/results/exp_0_baseline.csv \
    [--tta]
```

For each test image:
1. Load model & test split.
2. Predict (optionally with TTA).
3. Compute `metrics.compute_metrics(pred, true, mdist=3)`.
4. Record per-image F1, F1-integral, mean-euclidean, and inference time.
5. Append to CSV.

---

## 4. SLURM execution

### 4.1 Directory layout

```
experiments/
  generate_configs.py        # writes configs/
  train.py                   # thin wrapper: load yaml, call run_experiment
  evaluate.py                # model + dataset -> metrics CSV
  summarize.py               # collects all CSVs into one table
  configs/                   # generated YAML per experiment
  models/                    # saved .h5 per experiment
  results/                   # per-experiment CSV
  slurm/
    train_array.sbatch       # SLURM array job for training
    eval_array.sbatch        # SLURM array job for evaluation
    summarize.sbatch         # final summary (depends on eval)
```

### 4.2 Training array job -- `slurm/train_array.sbatch`

```bash
#!/usr/bin/env bash
#SBATCH --job-name=db-perf
#SBATCH --array=0-14
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/logs/train_%a.out
#SBATCH --error=experiments/logs/train_%a.err

module load cuda/12.x  # adjust to cluster

source activate deepblink  # or: source /path/to/venv/bin/activate

# Map array index to config file
CONFIGS=(experiments/configs/exp_*.yaml)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=$(basename "$CONFIG" .yaml)

echo "=== Training $EXP_NAME on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python experiments/train.py \
    --config "$CONFIG" \
    --savedir experiments/models/ \
    --run-name "$EXP_NAME"
```

### 4.3 Evaluation array job -- `slurm/eval_array.sbatch`

```bash
#!/usr/bin/env bash
#SBATCH --job-name=db-eval
#SBATCH --array=0-14
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/logs/eval_%a.out
#SBATCH --error=experiments/logs/eval_%a.err
#SBATCH --dependency=afterok:${TRAIN_JOB_ID}

module load cuda/12.x
source activate deepblink

MODELS=(experiments/models/exp_*.h5)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
EXP_NAME=$(basename "$MODEL" .h5)

# For TTA experiments, add --tta flag
TTA_FLAG=""
if [[ "$EXP_NAME" == *"tta"* ]]; then
    TTA_FLAG="--tta"
fi

python experiments/evaluate.py \
    --model "$MODEL" \
    --dataset "$DATASET_PATH" \
    --output "experiments/results/${EXP_NAME}.csv" \
    $TTA_FLAG
```

### 4.4 Summary job -- `slurm/summarize.sbatch`

```bash
#!/usr/bin/env bash
#SBATCH --job-name=db-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --dependency=afterok:${EVAL_JOB_ID}

python experiments/summarize.py \
    --results-dir experiments/results/ \
    --output experiments/summary.csv
```

### 4.5 One-liner to launch everything

```bash
# Generate all config files
python experiments/generate_configs.py --dataset /path/to/dataset.npz

# Submit training array, capture job ID
TRAIN_JOB_ID=$(sbatch --parsable slurm/train_array.sbatch)

# Submit eval array, depends on training
EVAL_JOB_ID=$(sbatch --parsable --export=TRAIN_JOB_ID=$TRAIN_JOB_ID slurm/eval_array.sbatch)

# Submit summary, depends on eval
sbatch --export=EVAL_JOB_ID=$EVAL_JOB_ID slurm/summarize.sbatch
```

Or wrap it all in a `run_all.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:?Usage: ./run_all.sh /path/to/dataset.npz}

python experiments/generate_configs.py --dataset "$DATASET"

TRAIN=$(sbatch --parsable experiments/slurm/train_array.sbatch)
echo "Training job: $TRAIN"

EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN \
    --export=ALL,DATASET_PATH="$DATASET" \
    experiments/slurm/eval_array.sbatch)
echo "Eval job: $EVAL"

sbatch --dependency=afterok:$EVAL experiments/slurm/summarize.sbatch
echo "Summary will run after eval completes."
```

---

## 5. Expected output

`experiments/summary.csv`:

```
experiment,f1_3px_mean,f1_3px_std,f1_integral_mean,euclidean_mean,train_time_min,infer_ms_per_img,gpu_mem_mb
baseline,0.85,0.03,0.72,1.12,45,18,2100
aug_all,0.89,0.02,0.78,0.95,50,18,2100
lr_cosine,...
...
combined_best_tta,...
```

`experiments/summarize.py` will also print a formatted comparison table
to stdout and optionally generate a bar chart (matplotlib) saved as
`experiments/summary.png`.

---

## 6. Implementation order

1. **Scaffold**: Create `experiments/` directory, `generate_configs.py`,
   `train.py`, `evaluate.py`, `summarize.py`.
2. **Focal loss**: Add `combined_focal_rmse` to `losses.py` + register in
   `__init__` and `io.py` custom objects.
3. **LR schedules + early stopping**: Add `_build_callbacks` to
   `training.py`, read new optional keys from config.
4. **Mixed precision**: Single flag check at top of `run_experiment`.
5. **TTA**: Add `predict_tta` to `inference.py`.
6. **Config generator**: Write all 15 YAML files programmatically.
7. **SLURM scripts**: Write the three `.sbatch` files + `run_all.sh`.
8. **Dry run locally**: Train experiment 0 for 2 epochs to verify the full
   pipeline end-to-end.
9. **Submit to cluster**.

---

## 7. Cluster requirements

- 1 GPU per experiment (15 GPUs simultaneously, or queued).
- ~32 GB RAM, 4 CPU cores per training job.
- Estimated **4 hours** per training run (200 epochs), **1 hour** for all evaluations.
- Total wall-clock if all 15 run in parallel: ~5 hours.
- Total GPU-hours: ~60-75h (15 trains x 4h + 15 evals x 1h).

---

## 8. Decisions to make before starting

- [ ] **Which dataset?** Pick one `.npz` file for all experiments. Ideally a
      medium-difficulty one representative of real use cases.
- [ ] **Epochs**: Keep 200 or reduce for faster iteration (e.g., 100 with
      early stopping)?
- [ ] **GPU type**: A100/V100/other -- affects mixed-precision speedup.
- [ ] **Wandb**: Enable for richer tracking, or keep off for simplicity?
- [ ] **Additional experiments?** e.g., deeper network (`ndown=3`), smaller
      `cell_size=2`, different block types (`inception`, `residual`).
