#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EPOCHS="${EPOCHS:-100}"
SHOTS="${SHOTS:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-python}"
LR="${LR:-0.001}"


# Execution device: 'auto' (prefer CUDA), 'cuda', or 'cpu'.
export ACCELERATOR="${ACCELERATOR:-auto}"


export PYTHONPATH="${PYTHONPATH:-}:src"

for dataset in cifar100 flowers102; do
  for method in zero_shot linear full lora bitfit; do
    "$PYTHON" -m vit_fewshot.train \
      --dataset "$dataset" \
      --method "$method" \
      --data-dir "$DATA_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --max-epochs "$EPOCHS" \
      --shots-per-class "$SHOTS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --seed "$SEED" \
      --learning-rate "$LR"
  done
done

"$PYTHON" -m vit_fewshot.analyze --results-dir "$OUTPUT_DIR" --seed "$SEED"
