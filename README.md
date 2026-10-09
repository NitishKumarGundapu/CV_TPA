# Few-shot ViT adaptation

Reproducible PyTorch Lightning pipeline for TPA-17. It compares a pretrained
ViT-B/16 on FC100-style CIFAR-100 and Flowers102 using:

- original pretrained ImageNet classifier ("zero-shot" per project definition)
- linear probing
- full fine-tuning
- LoRA on attention QKV projections
- BitFit

The pipeline reports held-out accuracy, macro F1, trainable parameters,
training time, and peak GPU memory. It also creates comparison plots and t-SNE
plots of test representations.

## Setup

```bash
conda activate base
python -m pip install -r requirements.txt
```

## Run everything

```bash
EPOCHS=10 SHOTS=5 BATCH_SIZE=32 bash scripts/run_all.sh
```

Datasets and pretrained weights download automatically. Results are written to
`outputs/<dataset>/<method>/seed_<seed>/`; aggregate files go to
`outputs/analysis/`.

Run one experiment:

```bash
PYTHONPATH=src python -m vit_fewshot.train \
  --dataset flowers102 --method lora --shots-per-class 5 --max-epochs 10
```

Useful controls:

```bash
DATA_DIR=./data OUTPUT_DIR=./outputs NUM_WORKERS=4 SEED=42 \
  EPOCHS=10 SHOTS=5 BATCH_SIZE=32 bash scripts/run_all.sh
```

`zero_shot` intentionally keeps the original 1,000-class ImageNet head. Since
ImageNet IDs do not match CIFAR-100 or Flowers102 IDs, its predictions are
saved but accuracy and F1 are reported as unavailable rather than as misleading
numbers.
