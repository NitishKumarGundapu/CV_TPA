#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# 1. PROJECT SETUP
# ============================================================

# Find the project root directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# Conda environment
CONDA_ENV="${CONDA_ENV:-torch_env}"


# ============================================================
# 2. MODEL AND DATASET CONFIGURATION
# ============================================================

MODEL_NAME="${MODEL_NAME:-vit_base_patch16_dinov3.lvd1689m}"

DATA_ROOT="${DATA_ROOT:-data}"
DATASET="${DATASET:-flowers102}"

# 0 means use all available training samples.
SHOTS_PER_CLASS="${SHOTS_PER_CLASS:-0}"

# Fraction of each class's test images held out for validation.
VAL_FRACTION="${VAL_FRACTION:-0.2}"

SEED="${SEED:-42}"


# ============================================================
# 3. TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"

EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"


# ============================================================
# 4. LoRA CONFIGURATION
# ============================================================

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"


# ============================================================
# 5. EXPERIMENT METHODS
# ============================================================

# Available methods:
# full   : Full model fine-tuning
# lp     : Linear probing
# lora   : Low-Rank Adaptation
# bitfit : Fine-tuning only bias parameters
# ncm    : Nearest Class Mean classifier

METHODS="${METHODS:-full lp lora bitfit ncm}"

# Convert method names into an array.
read -r -a METHOD_LIST <<< "${METHODS}"


# ============================================================
# 6. OUTPUT AND EVALUATION CONFIGURATION
# ============================================================

LOG_DIR="${LOG_DIR:-src/dino_v3/logs}"
RESULTS_DIR="${RESULTS_DIR:-results/dino_v3}"

TSNE_SAMPLES="${TSNE_SAMPLES:-1500}"
GRAD_CAM_SAMPLES="${GRAD_CAM_SAMPLES:-8}"

# 0 means evaluate all batches.
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"

# 1.0 means use the complete dataset.
# Smaller values can be used for quick experiments.
LIMIT_BATCHES="${LIMIT_BATCHES:-1.0}"


# ============================================================
# 7. PYTHON ENVIRONMENT
# ============================================================

# Use the specified Conda environment if Conda is available.
# Otherwise, use the system Python.

if command -v conda >/dev/null 2>&1; then
    PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
    PYTHON=(python)
fi


# ============================================================
# 8. OPTIONAL ARGUMENTS
# ============================================================

# Skip experiments that have already been completed.

SKIP_ARGS=()

if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
    SKIP_ARGS+=(--skip-existing)
fi


# ============================================================
# 9. RUN DINOv3 EXPERIMENT SWEEP
# ============================================================

"${PYTHON[@]}" -m src.dino_v3.main sweep \
    --model-name "${MODEL_NAME}" \
    --data-root "${DATA_ROOT}" \
    --dataset "${DATASET}" \
    --shots-per-class "${SHOTS_PER_CLASS}" \
    --val-fraction "${VAL_FRACTION}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --log-dir "${LOG_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --tsne-samples "${TSNE_SAMPLES}" \
    --grad-cam-samples "${GRAD_CAM_SAMPLES}" \
    --max-eval-batches "${MAX_EVAL_BATCHES}" \
    --limit-batches "${LIMIT_BATCHES}" \
    --methods "${METHOD_LIST[@]}" \
    "${SKIP_ARGS[@]}" \
    "$@"
