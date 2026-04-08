#!/usr/bin/env bash
#SBATCH --account=fly-image
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name="db-perf-single"
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=./experiments/logs/%x-%j.log
#SBATCH --partition=gpu_short
#SBATCH --time=04:00:00

# ==============================================================================
# deepBlink: Run a single performance experiment
#
# Usage (interactive):
#   ./experiments/run_single.sh [EXP_ID] [DATASET_NAME]
#
# Usage (SLURM):
#   sbatch experiments/run_single.sh [EXP_ID] [DATASET_NAME]
#
# EXP_ID defaults to 0 (baseline). See generate_configs.py for all IDs.
# DATASET_NAME defaults to "particle" (smallest dataset, fastest iteration).
#
# Datasets are auto-downloaded from figshare if not already cached.
# ==============================================================================

set -euo pipefail

EXP_ID=${1:-0}
DATASET_NAME=${2:-particle}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="${REPO_DIR}/experiments/datasets"

cd "$REPO_DIR"

# ── Download dataset if needed ───────────────────────────────────────────────
declare -A DATASET_URLS=(
    [microtubule]="https://ndownloader.figshare.com/files/25308635"
    [receptor]="https://ndownloader.figshare.com/files/25308641"
    [vesicle]="https://ndownloader.figshare.com/files/25308833"
    [particle]="https://ndownloader.figshare.com/files/25737140"
    [smfish]="https://ndownloader.figshare.com/files/25737254"
    [suntag]="https://ndownloader.figshare.com/files/25737257"
)

if [[ ! -v "DATASET_URLS[$DATASET_NAME]" ]]; then
    echo "ERROR: Unknown dataset '$DATASET_NAME'"
    echo "Available: ${!DATASET_URLS[*]}"
    exit 1
fi

DATASET="${CACHE_DIR}/${DATASET_NAME}.npz"
mkdir -p "$CACHE_DIR"

if [[ ! -f "$DATASET" ]]; then
    echo "=== Downloading ${DATASET_NAME}.npz from figshare ==="
    curl -L -o "$DATASET" "${DATASET_URLS[$DATASET_NAME]}"
    echo "  saved to $DATASET ($(du -h "$DATASET" | cut -f1))"
else
    echo "=== Using cached ${DATASET_NAME}.npz ($(du -h "$DATASET" | cut -f1)) ==="
fi

# ── Environment setup ────────────────────────────────────────────────────────
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
elif command -v conda &>/dev/null && conda env list | grep -q deepblink; then
    source activate deepblink
fi

echo ""
echo "============================================"
echo "  deepBlink Performance Experiment"
echo "  Experiment: $EXP_ID"
echo "  Dataset:    $DATASET_NAME"
echo "  Python:     $(python --version)"
echo "  Date:       $(date)"
echo "  Node:       $(hostname)"
echo "============================================"

# GPU check
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs: {len(gpus)}')
for g in gpus: print(f'  {g}')
" 2>/dev/null || true

# ── Generate config ──────────────────────────────────────────────────────────
echo ""
echo "=== Generating config ==="
python experiments/generate_configs.py --dataset "$DATASET" --experiment "$EXP_ID"

CONFIG=$(ls experiments/configs/exp_$(printf '%02d' "$EXP_ID")_*.yaml)
EXP_NAME=$(basename "$CONFIG" .yaml)

# ── Train ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Training $EXP_NAME ==="
python experiments/train.py --config "$CONFIG"

# ── Evaluate ─────────────────────────────────────────────────────────────────
MODEL=$(ls -t experiments/models/*"${EXP_NAME}"*.h5 2>/dev/null | head -1)
if [[ -z "$MODEL" ]]; then
    echo "ERROR: No model found for $EXP_NAME"
    exit 1
fi

echo ""
echo "=== Evaluating $EXP_NAME ==="
TTA_FLAG=""
if [[ "$EXP_NAME" == *"tta"* ]]; then
    TTA_FLAG="--tta"
fi

python experiments/evaluate.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output "experiments/results/${EXP_NAME}.csv" \
    $TTA_FLAG

echo ""
echo "=== Done: $EXP_NAME ($(date)) ==="
