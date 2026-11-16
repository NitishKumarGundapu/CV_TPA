#!/usr/bin/env bash
set -euo pipefail

# Run the full DINOv3 adaptation sweep from the repository root.
# Usage: ./src/dino_v3/run.sh [extra main.py flags]
# Example: METHODS="lora bitfit" EPOCHS=10 ./src/dino_v3/run.sh --skip-existing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

CONDA_ENV="${CONDA_ENV:-torch_env}"

MODEL_NAME="${MODEL_NAME:-vit_base_patch16_dinov3.lvd1689m}"
DATA_ROOT="${DATA_ROOT:-data}"
SHOTS_PER_CLASS="${SHOTS_PER_CLASS:-10}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-10}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LOG_DIR="${LOG_DIR:-src/dino_v3/logs}"
RESULTS_DIR="${RESULTS_DIR:-results/dino_v3}"
TSNE_SAMPLES="${TSNE_SAMPLES:-1500}"
GRAD_CAM_SAMPLES="${GRAD_CAM_SAMPLES:-8}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
LIMIT_BATCHES="${LIMIT_BATCHES:-1.0}"
METHODS="${METHODS:-full lp lora bitfit ncm}"

read -r -a METHOD_LIST <<< "${METHODS}"

if command -v conda >/dev/null 2>&1; then
  PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  PYTHON=(python)
fi

SKIP_ARGS=()
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  SKIP_ARGS+=(--skip-existing)
fi

"${PYTHON[@]}" -m src.dino_v3.main sweep \
  --model-name "${MODEL_NAME}" \
  --data-root "${DATA_ROOT}" \
  --shots-per-class "${SHOTS_PER_CLASS}" \
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
