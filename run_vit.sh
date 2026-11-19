#!/usr/bin/env bash
set -euo pipefail

# Run all ViT adaptation methods for 10 epochs on the full training split of
# both datasets, then evaluate every method on the full test split.
#
#   results:  results/vit/<dataset>/<method>/seed_<seed>/
#   analysis: results/analysis/<dataset>/vit/
#
# Usage: bash run_vit.sh

cd "$(dirname "${BASH_SOURCE[0]}")"

CONDA_ENV="${CONDA_ENV:-torch_env}"
DATASETS=(flowers102 oxfordiiitpet)
METHODS=(full lp lora bitfit ncm)

if command -v conda >/dev/null 2>&1; then
    PYTHON=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
    PYTHON=(python)
fi

for DATASET in "${DATASETS[@]}"; do
    echo "=== ViT / ${DATASET}: methods=${METHODS[*]} epochs=10 batch=32 full-train ==="
    "${PYTHON[@]}" -m src.vit.main sweep \
        --dataset "${DATASET}" \
        --methods "${METHODS[@]}" \
        --shots-per-class 0 \
        --val-fraction 0.2 \
        --epochs 10 \
        --batch-size 32 \
        --data-root data \
        --results-dir results/vit \
        --log-dir src/vit/logs
done

"${PYTHON[@]}" build_analysis.py
