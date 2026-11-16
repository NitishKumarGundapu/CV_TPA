# DINOv3 Few-Shot Flowers102 Pipeline

Fine-tuning the DINOv3 foundation model for few-shot image classification,
mirroring the adaptation strategies in `src/vit` on the same Flowers102 split.

## Methods

| method | what is trained |
| --- | --- |
| `full` | every backbone and head parameter |
| `lp` | classification head only (linear probe) |
| `lora` | LoRA adapters on attention `qkv`/`proj` + head |
| `bitfit` | all bias terms + head |
| `ncm` | nothing (nearest-class-mean prototypes on frozen CLS features) |

`peft` is an alias for `lora`, `zs` for `ncm`, and `linear` for `lp`.

DINOv3 is a self-supervised vision backbone without a text encoder, so true
text-prompt zero-shot classification is not possible. `ncm` is the honest
training-free few-shot baseline: it builds class prototypes from the labeled
shots and classifies by cosine similarity, with no optimizer and zero trainable
parameters.

## Model and data

The default backbone is `vit_base_patch16_dinov3.lvd1689m`, loaded through
`timm` from the ungated `timm/vit_base_patch16_dinov3.lvd1689m` checkpoint.
Preprocessing (256x256 input, bicubic, ImageNet normalization) is resolved from
the timm pretrained configuration.

Flowers102 is downloaded by `torchvision` into `data/flowers-102/`. The official
training split (10 images/class) is subsampled deterministically with
`--shots-per-class`; validation and test use the full official splits (1020 and
6149 images).

## Setup

```bash
conda activate torch_env
pip install -r requirements.txt
```

## Usage

Run from the repository root.

```bash
# Full comparison: train every strategy, evaluate, and write the report
python -m src.dino_v3.main sweep --methods full lp lora bitfit ncm

# One strategy
python -m src.dino_v3.main train --method lora --shots-per-class 10 --epochs 30

# Re-evaluate an existing checkpoint
python -m src.dino_v3.main evaluate \
    --checkpoint src/dino_v3/logs/dinov3_classifier_lora/version_0/checkpoints/best-29-0.13.ckpt

# Rebuild the comparison report
python -m src.dino_v3.main report

# Fast smoke test
python -m src.dino_v3.main sweep --methods lp ncm --shots-per-class 2 --epochs 1 \
    --limit-batches 2 --max-eval-batches 3 --grad-cam-samples 2
```

## Inference

`predict` runs on individual images or a directory using either a trained
checkpoint or the `prototypes.npz` written by an NCM run.

```bash
# From a trained checkpoint
python -m src.dino_v3.main predict \
    --checkpoint src/dino_v3/logs/dinov3_classifier_lora/version_0/checkpoints/best-29-0.13.ckpt \
    --images data/flowers-102/jpg/image_00001.jpg --top-k 5

# From NCM prototypes
python -m src.dino_v3.main predict \
    --prototypes results/dino_v3/flowers102/ncm/seed_42/prototypes.npz \
    --image-dir some/folder --output predictions.csv
```

Each result contains the predicted class id (0-101), confidence, and top-k
class scores.

Useful flags: `--seed`, `--batch-size`, `--num-workers`, `--patience`,
`--lora-r`, `--lora-alpha`, `--lora-dropout`, `--tsne-samples`,
`--grad-cam-samples`, `--max-eval-batches`, `--limit-batches`, `--skip-existing`.

## Outputs

Each run writes to `results/dino_v3/flowers102/<method>/seed_<seed>/`:

- `metrics.json` — top-1, top-5, macro/weighted F1, parameter counts, training
  time, peak GPU memory, best validation loss, checkpoint path, and notes
- `predictions.npz` — logits, labels, and CLS embeddings for validation and test
- `per_class_f1.csv`, `confusion_matrix.csv`, `figures/confusion_test.png`
- `figures/tsne_val.png`, `figures/gradcam_val.png`
- `prototypes.npz` — class prototypes plus model metadata for `ncm` runs

The sweep also writes `results/dino_v3/analysis/summary.csv`, `summary.md`,
`tradeoffs.png`, and `accuracy_pareto.png`.

## Project layout

```
src/dino_v3/
├── config.py         # Run configuration and CLI argument definitions
├── dataset.py        # Flowers102FewShot and timm DINOv3 transforms
├── model.py          # DINOv3Classifier LightningModule and adaptation methods
├── metrics.py        # Top-k accuracy, F1, per-class report, confusion matrix
├── evaluate.py       # Inference, NCM prototypes, metrics/artifacts, prediction
├── visualization.py  # t-SNE and Grad-CAM
├── report.py         # Cross-method tables and plots
└── main.py           # train / evaluate / predict / sweep / report CLI
```
