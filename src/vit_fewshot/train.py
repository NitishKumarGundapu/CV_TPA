from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
import warnings
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import accuracy_score, f1_score

from data import DATASET_CLASSES, FewShotDataModule
from model_timm import METHODS, ViTClassifier

torch.set_float32_matmul_precision('medium')

warnings.filterwarnings(
    "ignore", 
    message=".*`isinstance\\(treespec, LeafSpec\\)` is deprecated.*", 
    category=FutureWarning
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot ViT adaptation")
    parser.add_argument("--dataset", default="flowers102")
    parser.add_argument("--method", default="full")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--shots-per-class", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    print(f"[device] using accelerator '{device}'", file=sys.stderr)
    L.seed_everything(args.seed, workers=True)
    run_dir = (
        Path(args.output_dir) / args.dataset / args.method / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    data = FewShotDataModule(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shots_per_class=args.shots_per_class,
        seed=args.seed,
    )

    data.prepare_data()
    data.setup()

    model = ViTClassifier(
        num_classes=DATASET_CLASSES[args.dataset],
        method=args.method,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
    )

    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor="val/accuracy",
        mode="max",
        save_top_k=1,
        filename="best",
    )

    logger = CSVLogger(save_dir=run_dir, name="logs", version=".")
    
    trainer = L.Trainer(
        limit_train_batches = 1000,
        limit_val_batches = 200,
        accelerator = device,
        max_epochs = args.max_epochs,
        logger = logger,
        callbacks=[] if args.method == "zero_shot" else [checkpoint],
        enable_checkpointing = args.method != "zero_shot",
        deterministic = True,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()

    trainer.fit(model, data.train_dataloader(), data.val_dataloader())

    training_seconds = time.perf_counter() - started

    trainer.test(
        model,
        datamodule=data,
        ckpt_path="best" if args.method != "zero_shot" else None,
    )


    ######## Predict and save metrics ########

    predictions = torch.cat(model.test_predictions).numpy()
    targets = torch.cat(model.test_targets).numpy()
    compatible_labels = model.output_classes == model.num_classes
    metrics = {
        "dataset": args.dataset,
        "method": args.method,
        "seed": args.seed,
        "shots_per_class": args.shots_per_class,
        "accuracy": float(accuracy_score(targets, predictions))
        if compatible_labels
        else None,
        "macro_f1": float(f1_score(targets, predictions, average="macro"))
        if compatible_labels
        else None,
        "trainable_parameters": model.trainable_parameters,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": training_seconds,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        ),
        "prediction_label_space": (
            args.dataset if compatible_labels else "imagenet-1k"
        ),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        predictions=predictions,
        targets=targets
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()