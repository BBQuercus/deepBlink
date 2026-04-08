#!/usr/bin/env bash
#SBATCH --account=fly-image
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name="db-perf-all"
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=./experiments/logs/%x-%j.log
#SBATCH --partition=main
#SBATCH --time=12:00:00

# ==============================================================================
# deepBlink: Run ALL performance experiments on a single dataset
#
# Usage (interactive):
#   ./experiments/run_all.sh [DATASET_NAME]
#
# Usage (SLURM):
#   sbatch experiments/run_all.sh [DATASET_NAME]
#
# DATASET_NAME defaults to "particle". Datasets auto-download from figshare.
# ==============================================================================

set -euo pipefail

DATASET_NAME=${1:-particle}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="/tachyon/scratch/gchao/douyuhu/deepblink_datasets"

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
echo "  deepBlink Performance Experiments"
echo "  Dataset:  $DATASET_NAME"
echo "  Python:   $(python --version)"
echo "  Date:     $(date)"
echo "  Node:     $(hostname)"
echo "============================================"

python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs: {len(gpus)}')
for g in gpus: print(f'  {g}')
" 2>/dev/null || true

# ── Generate all configs ─────────────────────────────────────────────────────
echo ""
echo "=== Generating all configs ==="
python experiments/generate_configs.py --dataset "$DATASET"

# ── Run experiments sequentially ─────────────────────────────────────────────
FAILED=()
for CONFIG in experiments/configs/exp_*.yaml; do
    EXP_NAME=$(basename "$CONFIG" .yaml)
    echo ""
    echo "================================================================"
    echo "=== $EXP_NAME ($(date)) ==="
    echo "================================================================"

    # Train
    if ! python experiments/train.py --config "$CONFIG"; then
        echo "FAILED: $EXP_NAME (training)"
        FAILED+=("$EXP_NAME")
        continue
    fi

    # Find model
    MODEL=$(ls -t experiments/models/*"${EXP_NAME}"*.h5 2>/dev/null | head -1)
    if [[ -z "$MODEL" ]]; then
        echo "FAILED: $EXP_NAME (no model saved)"
        FAILED+=("$EXP_NAME")
        continue
    fi

    # Evaluate
    TTA_FLAG=""
    if [[ "$EXP_NAME" == *"tta"* ]]; then
        TTA_FLAG="--tta"
    fi

    python experiments/evaluate.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output "experiments/results/${EXP_NAME}.csv" \
        $TTA_FLAG

    echo "=== Done: $EXP_NAME ==="
done

# ── Summarize ────────────────────────────────────────────────────────────────
echo ""
echo "=== Generating summary ==="
python experiments/summarize.py

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: ${#FAILED[@]} experiment(s) failed: ${FAILED[*]}"
fi
