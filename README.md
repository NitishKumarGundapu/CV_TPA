# TPA-17: Few-shot Flowers102 classification with a ViT foundation model

Fine-tuning a pretrained Vision Transformer for image classification with few
labeled samples, comparing adaptation strategies on accuracy, parameter count,
and compute cost.

## Implemented strategies

| method | what is trained | trainable params (ViT-B/16, 102 classes) |
| --- | --- | --- |
| `full` | every parameter | 85.9 M (100%) |
| `lp` | classification head only (linear probe) | 78.4 K (0.09%) |
| `lora` | LoRA adapters on `q_proj`/`v_proj` + head | 668.3 K (0.77%) |
| `bitfit` | all bias terms + head | 181.4 K (0.21%) |
| `ncm` | nothing (nearest-class-mean on frozen CLS features) | 0 (0%) |

`peft` is accepted as an alias for `lora`, and `zs` as an alias for `ncm`.

### Note on zero-shot

A plain ViT classification head has no text/image alignment, so true
text-prompt zero-shot classification is not possible without a vision-language
model such as CLIP. The previous `zs` path loaded the 1000-class ImageNet head
and computed cross-entropy against 102 flower labels, which produced a
meaningless `accuracy = 0.0`. That path has been replaced by `ncm`: a
training-free nearest-class-mean prototype classifier over frozen ViT features
(no optimizer, no gradient updates, zero trainable parameters). The cached
ImageNet head only contains two flower classes, and CLIP is out of scope for
this ViT-only setup, so `ncm` is the honest no-training baseline here. This
caveat is recorded in each `ncm` run's `metrics.json`.

## Setup

```bash
conda activate torch_env
pip install -r requirements.txt
```

The pipeline defaults to `google/vit-base-patch16-224`; weights are loaded from
the Hugging Face cache or downloaded on first use.

## Data

Flowers102 is downloaded automatically by `torchvision` into `data/`:

```
data/flowers-102/
├── jpg/                 # 8189 images
├── imagelabels.mat
└── setid.mat
```

The official training split (10 images/class) is subsampled deterministically
with `--shots-per-class`; validation and test always use the full official
splits (1020 and 6149 images).

## Usage

Run everything from the repository root with `python -m src.vit.main`.

```bash
# Full comparison: trains every strategy, evaluates, and writes the report
python -m src.vit.main sweep --methods full lp lora bitfit ncm

# One strategy
python -m src.vit.main train --method lora --shots-per-class 10 --epochs 30

# Re-evaluate an existing checkpoint (metrics, t-SNE, Grad-CAM, confusion matrix)
python -m src.vit.main evaluate --checkpoint src/vit/logs/vit_classifier_lora/version_0/checkpoints/best-29-0.13.ckpt

# Rebuild the comparison report from existing runs
python -m src.vit.main report

# Fast smoke test (~1 minute)
python -m src.vit.main sweep --methods lp ncm --shots-per-class 2 --epochs 1 \
    --limit-batches 2 --max-eval-batches 3 --grad-cam-samples 2
```

Useful flags: `--seed`, `--batch-size`, `--num-workers`, `--patience`,
`--lora-r`, `--lora-alpha`, `--lora-dropout`, `--tsne-samples`,
`--grad-cam-samples`, `--max-eval-batches`, `--limit-batches`, `--skip-existing`.

## Results (Flowers102, 10 shots/class, seed 42)

Measured on the official 6149-image test split with `google/vit-base-patch16-224`
(30 epochs, batch 16, RTX PRO 4000). Full table and plots:
`results/analysis/summary.md`, `tradeoffs.png`, `accuracy_pareto.png`.

| method | top-1 acc | top-5 acc | macro F1 | trainable params | % trainable | train time (s) | peak GPU (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | 0.9849 | 0.9980 | 0.9854 | 85,877,094 | 100.00 | 435.1 | 3182 |
| lp | 0.9867 | 0.9974 | 0.9850 | 78,438 | 0.09 | 217.3 | 512 |
| lora | 0.9912 | 0.9977 | 0.9899 | 668,262 | 0.77 | 320.7 | 1870 |
| bitfit | 0.9915 | 0.9972 | 0.9907 | 181,350 | 0.21 | 311.1 | 1596 |
| ncm | 0.9859 | 0.9971 | 0.9833 | 0 | 0.00 | 6.2 | 489 |

Observations:

- All strategies are within 0.7 points of each other: this dataset is saturated
  by the pretrained ViT representation, so the interesting axis is efficiency,
  not raw accuracy.
- BitFit reaches the best top-1 (99.15%) while training 0.21% of the parameters,
  and LoRA is within 0.03 points at 0.77%. Full fine-tuning is not better here
  and costs 3.2 GB / 435 s.
- Linear probing is the cheapest trained model (78 K parameters, 217 s, 512 MB)
  and already reaches 98.67%.
- `ncm` is training-free and gets 98.59% from class means of frozen CLS features,
  which underscores how linearly separable these features already are. It uses
  the 10 labeled examples per class to form prototypes, so it is a few-shot
  prototype baseline, not text-prompt zero-shot.

Per-run artifacts (t-SNE, Grad-CAM, confusion matrix, per-class F1, raw
predictions) live under `results/flowers102/<method>/seed_42/`.

## Evaluation outputs

Each run writes to `results/flowers102/<method>/seed_<seed>/`:

- `metrics.json` — top-1, top-5, macro/weighted F1, trainable/total parameters,
  training wall-clock seconds, peak GPU memory, best validation loss, checkpoint
  path, and notes
- `predictions.npz` — logits, labels, and CLS embeddings for validation and test
- `per_class_f1.csv` — precision/recall/F1/support for all 102 classes
- `confusion_matrix.csv` and `figures/confusion_test.png`
- `figures/tsne_val.png` — t-SNE of validation CLS embeddings
- `figures/gradcam_val.png` — Grad-CAM from the last ViT block
- `prototypes.npz` — class prototypes for `ncm` runs

The sweep also writes `results/analysis/summary.csv`, `summary.md`,
`tradeoffs.png` (accuracy / macro-F1 / trainable-parameter bars), and
`accuracy_pareto.png` (accuracy vs. parameters and training time).

Training curves remain in `src/vit/logs/vit_classifier_<method>/<version>/metrics.csv`.

## Reproducibility

```bash
python -m src.vit.main sweep --methods full lp lora bitfit ncm --seed 42
```

`pl.seed_everything(seed, workers=True)` seeds Python, NumPy, PyTorch, and the
data loaders; the few-shot subset is sampled with a seeded `torch.Generator`.
Metrics are computed on the official test split, and the best checkpoint is
selected by validation loss (early stopping, patience 10).

## Project layout

```
src/vit/
├── config.py         # Run configuration and CLI argument definitions
├── dataset.py        # Flowers102FewShot dataset and dataloaders
├── trainer.py        # ViTClassifier LightningModule and adaptation methods
├── metrics.py        # Top-k accuracy, F1, per-class report, confusion matrix
├── evaluate.py       # Inference, NCM prototypes, metric/artifact saving
├── visualization.py  # t-SNE and Grad-CAM
├── report.py         # Cross-method tables and plots
└── main.py           # train / evaluate / sweep / report CLI
```
