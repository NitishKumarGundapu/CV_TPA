#!/usr/bin/env bash
set -euo pipefail

# Run all DINOv3 adaptation methods for 20 epochs with few-shot training
# (flowers102: 5 shots/class, oxfordiiitpet: 10 shots/class), then evaluate
# every method on the full test split.
#
#   results:  results/dino_v3/<dataset>/<method>/seed_<seed>/
#   analysis: results/analysis/<dataset>/dinov3/
#
# Usage: bash run_dino.sh

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
    case "${DATASET}" in
        flowers102) SHOTS_PER_CLASS=5 ;;
        oxfordiiitpet) SHOTS_PER_CLASS=10 ;;
        *)
            echo "Unknown dataset: ${DATASET}" >&2
            exit 1
            ;;
    esac

    echo "=== DINOv3 / ${DATASET}: methods=${METHODS[*]} epochs=20 batch=32 shots-per-class=${SHOTS_PER_CLASS} ==="
    "${PYTHON[@]}" -m src.dino_v3.main sweep \
        --dataset "${DATASET}" \
        --methods "${METHODS[@]}" \
        --shots-per-class "${SHOTS_PER_CLASS}" \
        --val-fraction 0.2 \
        --epochs 20 \
        --batch-size 32 \
        --data-root data \
        --results-dir results/dino_v3 \
        --log-dir logs/dino_v3/
done

"${PYTHON[@]}" build_analysis.py
