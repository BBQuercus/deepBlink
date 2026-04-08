#!/usr/bin/env bash
#SBATCH --account=fly-image
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name="db-retrain"
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=./experiments/logs/%x-%j.log
#SBATCH --partition=main
#SBATCH --time=12:00:00

# ==============================================================================
# deepBlink: Retrain all models with current TF/Keras (clean, paper settings)
#
# Produces publish-ready .h5 models for all 6 original paper datasets.
# Datasets are auto-downloaded from figshare.
#
# Usage (interactive):
#   ./experiments/retrain_clean.sh
#
# Usage (SLURM):
#   sbatch experiments/retrain_clean.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="/tachyon/scratch/gchao/douyuhu/deepblink_datasets"
OUTDIR="${REPO_DIR}/experiments/retrained_models"

cd "$REPO_DIR"

# ── Figshare dataset URLs ────────────────────────────────────────────────────
declare -A DATASET_URLS=(
    [microtubule]="https://ndownloader.figshare.com/files/25308635"
    [receptor]="https://ndownloader.figshare.com/files/25308641"
    [vesicle]="https://ndownloader.figshare.com/files/25308833"
    [particle]="https://ndownloader.figshare.com/files/25737140"
    [smfish]="https://ndownloader.figshare.com/files/25737254"
    [suntag]="https://ndownloader.figshare.com/files/25737257"
)

MODELS=(microtubule receptor vesicle particle smfish suntag)

# ── Download all datasets ────────────────────────────────────────────────────
mkdir -p "$CACHE_DIR" "$OUTDIR"

echo "=== Downloading datasets from figshare ==="
for name in "${MODELS[@]}"; do
    DATASET="${CACHE_DIR}/${name}.npz"
    if [[ ! -f "$DATASET" ]]; then
        echo "  downloading ${name}.npz..."
        curl -L -o "$DATASET" "${DATASET_URLS[$name]}"
        echo "    saved ($(du -h "$DATASET" | cut -f1))"
    else
        echo "  cached ${name}.npz ($(du -h "$DATASET" | cut -f1))"
    fi
done

# ── Environment setup ────────────────────────────────────────────────────────
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
elif command -v conda &>/dev/null && conda env list | grep -q deepblink; then
    source activate deepblink
fi

echo ""
echo "============================================"
echo "  deepBlink Clean Retrain (paper settings)"
echo "  Python:  $(python --version)"
echo "  Date:    $(date)"
echo "  Node:    $(hostname)"
echo "  Output:  $OUTDIR"
echo "============================================"

python -c "
import tensorflow as tf, keras
print(f'TensorFlow: {tf.__version__}')
print(f'Keras:      {keras.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs:       {len(gpus)}')
for g in gpus: print(f'  {g}')
" 2>/dev/null || true

# ── Train all models (paper defaults) ────────────────────────────────────────
FAILED=()
for name in "${MODELS[@]}"; do
    DATASET="${CACHE_DIR}/${name}.npz"
    CONFIG="${OUTDIR}/${name}_config.yaml"

    echo ""
    echo "================================================================"
    echo "=== Training: ${name} ($(date)) ==="
    echo "================================================================"

    cat > "$CONFIG" <<YAML
name: deepBlink
run_name: ${name}
savedir: ${OUTDIR}
use_wandb: false

augmentation_args:
  flip: false
  illuminate: false
  rotate: false
  gaussian_noise: false
  translate: false

dataset: SpotsDataset
dataset_args:
  name: ${DATASET}
  cell_size: 4
  smooth_factor: 1

model: SpotsModel
network: unet
network_args:
  dropout: 0.3
  filters: 5
  ndown: 2
  l2: 1.0e-06
  block: convolutional

loss: combined_dice_rmse
optimizer: amsgrad

train_args:
  batch_size: 2
  epochs: 200
  learning_rate: 1.0e-04
  overfit: false
  pre_train: null
YAML

    if python experiments/train.py --config "$CONFIG"; then
        echo "=== Done: ${name} ($(date)) ==="
    else
        echo "FAILED: ${name}"
        FAILED+=("$name")
    fi
done

# ── Evaluate all retrained models ────────────────────────────────────────────
echo ""
echo "=== Evaluating retrained models ==="
mkdir -p "${OUTDIR}/eval"

for name in "${MODELS[@]}"; do
    MODEL=$(ls -t "${OUTDIR}"/*"${name}"*.h5 2>/dev/null | head -1)
    DATASET="${CACHE_DIR}/${name}.npz"

    if [[ -z "$MODEL" ]]; then
        echo "  skipping $name (no model found)"
        continue
    fi

    echo "  evaluating $name..."
    python experiments/evaluate.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output "${OUTDIR}/eval/${name}.csv"
done

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "=== Clean retrain complete ==="
echo "================================================================"
echo ""
echo "Models:"
ls -lh "${OUTDIR}"/*.h5 2>/dev/null || echo "  (none)"
echo ""

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "WARNING: ${#FAILED[@]} model(s) failed: ${FAILED[*]}"
else
    echo "All ${#MODELS[@]} models trained successfully."
fi

echo ""
echo "Next steps:"
echo "  1. Review eval results in ${OUTDIR}/eval/"
echo "  2. Upload .h5 files to figshare / HuggingFace"
echo "  3. Update deepblink/cli/models.txt with new URLs"
