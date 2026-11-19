# DINOv3 Few-Shot Flowers102 Pipeline

Fine-tuning the DINOv3 foundation model for few-shot image classification,
mirroring the adaptation strategies in `src/vit` on the same Flowers102 and
Oxford-IIIT Pet splits.

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

Flowers102 is downloaded by `torchvision` into `data/flowers-102/`. The full
official training split (10 images/class) is used by default; pass
`--shots-per-class` to subsample it deterministically. Validation and test use
the full official splits (1020 and 6149 images).

Oxford-IIIT Pet is available with `--dataset oxfordiiitpet`. It is loaded with
`torchvision.datasets.ImageFolder` from `data/oxford_pet/{train,test}/<class>/`.
Training uses the full `train` folder (37 classes, roughly 100 images/class)
unless `--shots-per-class` is set. There is no validation folder, so
`--val-per-class` (default 10) images per class are held out deterministically
from `test` for validation; the remaining test images form the test split.

## Results (CIFAR-100, 10 shots/class, seed 42)

10 epochs, batch 32, 1000-image validation split, official 10k test split,
RTX PRO 4000. Full/`lp` use per-method learning rates (2e-5 and 1e-3);
LoRA/BitFit use 1e-4.

| method | top-1 acc | top-5 acc | macro F1 | trainable params | % trainable | train time (s) | peak GPU (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 0.5802 | 0.8538 | 0.5604 | 85,718,116 | 100.00 | 201.8 | 7362 |
| lp | 0.8144 | 0.9627 | 0.8150 | 76,900 | 0.09 | 113.1 | 710 |
| lora | 0.5977 | 0.8563 | 0.5771 | 961,636 | 1.11 | 170.9 | 4908 |
| bitfit | 0.5106 | 0.7735 | 0.4954 | 152,164 | 0.18 | 159.1 | 4171 |
| ncm | 0.8164 | 0.9624 | 0.8173 | 0 | 0.00 | 14.9 | 702 |

At 10 shots/class the frozen DINOv3 features win: NCM and linear probing beat
full fine-tuning and LoRA, which overfit the small training set. DINOv3 full
fine-tuning is LR-sensitive; 1e-4 collapsed to 13.90% top-1 while 2e-5 reached
58.02%.

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

# Oxford-IIIT Pet comparison
python -m src.dino_v3.main sweep --dataset oxfordiiitpet --methods full lp lora bitfit ncm \
    --epochs 10 --patience 3 --batch-size 32

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

Each run writes to `results/dino_v3/<dataset>/<method>/seed_<seed>/`:

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
├── dataset.py        # Flowers102/Oxford-IIIT Pet datasets and timm DINOv3 transforms
├── model.py          # DINOv3Classifier LightningModule and adaptation methods
├── metrics.py        # Top-k accuracy, F1, per-class report, confusion matrix
├── evaluate.py       # Inference, NCM prototypes, metrics/artifacts, prediction
├── visualization.py  # t-SNE and Grad-CAM
├── report.py         # Cross-method tables and plots
└── main.py           # train / evaluate / predict / sweep / report CLI
```
