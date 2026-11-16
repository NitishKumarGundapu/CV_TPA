from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
import pytorch_lightning as pl
import torch

from .config import (
    METHODS,
    REPO_ROOT,
    Config,
    add_common_arguments,
    config_from_args,
    normalize_method,
    resolve_path,
)
from .dataset import build_dataloaders, build_datasets, build_transforms
from .evaluate import (
    collect_images,
    evaluate_and_save,
    extract_outputs,
    load_checkpoint,
    nearest_class_mean,
    predict_paths,
    save_predictions,
)
from .metrics import accuracy, top_k_accuracy
from .model import DINOv3Classifier
from .report import build_report

COMMON_FLAGS = [
    ("model_name", "--model-name"),
    ("data_root", "--data-root"),
    ("shots_per_class", "--shots-per-class"),
    ("seed", "--seed"),
    ("batch_size", "--batch-size"),
    ("num_workers", "--num-workers"),
    ("epochs", "--epochs"),
    ("patience", "--patience"),
    ("lr", "--lr"),
    ("weight_decay", "--weight-decay"),
    ("lora_r", "--lora-r"),
    ("lora_alpha", "--lora-alpha"),
    ("lora_dropout", "--lora-dropout"),
    ("log_dir", "--log-dir"),
    ("results_dir", "--results-dir"),
    ("tsne_samples", "--tsne-samples"),
    ("grad_cam_samples", "--grad-cam-samples"),
    ("max_eval_batches", "--max-eval-batches"),
    ("limit_batches", "--limit-batches"),
]

NCM_NOTE = (
    "Training-free nearest-class-mean over frozen DINOv3 CLS features (0 trainable "
    "parameters). DINOv3 is a self-supervised vision backbone without a text encoder, "
    "so text-prompt zero-shot classification is not possible; NCM over the few labeled "
    "shots is the training-free baseline."
)


def _common_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    for attribute, flag in COMMON_FLAGS:
        flags.extend([flag, str(getattr(args, attribute))])
    return flags


def _build_model(cfg: Config) -> DINOv3Classifier:
    return DINOv3Classifier(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        method=cfg.method,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        metric_fns={"acc": accuracy, "top5": top_k_accuracy},
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    )


def run_train(cfg: Config) -> dict:
    pl.seed_everything(cfg.seed, workers=True)
    train_dataset, train_eval_dataset, val_dataset, test_dataset = build_datasets(cfg)
    print(
        f"[data] {cfg.dataset}: train={len(train_dataset)} val={len(val_dataset)} "
        f"test={len(test_dataset)} shots_per_class={cfg.shots_per_class}"
    )
    loaders = build_dataloaders(
        cfg, train_dataset, train_eval_dataset, val_dataset, test_dataset
    )

    model = _build_model(cfg)
    counts = model.parameter_counts()
    print(
        f"[params] trainable={counts['trainable_parameters']:,} "
        f"total={counts['total_parameters']:,} "
        f"({100 * counts['trainable_fraction']:.2f}%)"
    )

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path: str | None = None
    best_val_loss: float | None = None
    prototypes = None
    notes = ""

    if cfg.method == "ncm":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        train_outputs = extract_outputs(
            model, loaders["train_eval"], device, cfg.max_eval_batches
        )
        prototypes = nearest_class_mean(
            train_outputs["embeddings"], train_outputs["labels"], cfg.num_classes
        )
        train_seconds = time.perf_counter() - started
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else None
        )
        notes = NCM_NOTE
        np.savez_compressed(
            cfg.run_dir / "prototypes.npz",
            prototypes=prototypes,
            model_name=np.array(cfg.model_name),
            num_classes=np.array(cfg.num_classes),
        )
    else:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger = pl.loggers.CSVLogger(
            str(resolve_path(cfg.log_dir)), name=f"dinov3_classifier_{cfg.method}"
        )
        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:02d}-{val/loss:.4f}",
            auto_insert_metric_name=False,
        )
        callbacks = [
            checkpoint,
            pl.callbacks.EarlyStopping(
                monitor="val/loss", patience=cfg.patience, mode="min"
            ),
            pl.callbacks.LearningRateMonitor("step"),
        ]
        trainer = pl.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="auto",
            devices="auto",
            logger=logger,
            callbacks=callbacks,
            limit_train_batches=cfg.limit_batches,
            limit_val_batches=cfg.limit_batches,
            log_every_n_steps=10,
        )
        started = time.perf_counter()
        trainer.fit(model, loaders["train"], loaders["val"])
        train_seconds = time.perf_counter() - started
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else None
        )
        best_val_loss = trainer.callback_metrics.get("val/loss")
        best_val_loss = float(best_val_loss) if best_val_loss is not None else None
        checkpoint_path = checkpoint.best_model_path
        if not checkpoint_path:
            raise RuntimeError("Training finished without writing a checkpoint")
        model = load_checkpoint(checkpoint_path)

    record = evaluate_and_save(
        model,
        cfg,
        {"val": loaders["val"], "test": loaders["test"]},
        train_seconds=train_seconds,
        peak_gpu_memory_mb=peak_gpu_memory_mb,
        best_val_loss=best_val_loss,
        checkpoint=checkpoint_path,
        prototypes=prototypes,
        notes=notes,
    )
    return record


def run_evaluate(args: argparse.Namespace) -> dict:
    cfg = config_from_args(args)
    model = load_checkpoint(args.checkpoint)
    cfg.method = model.method
    cfg.model_name = model.model_name
    cfg.num_classes = model.num_classes

    train_dataset, train_eval_dataset, val_dataset, test_dataset = build_datasets(cfg)
    loaders = build_dataloaders(
        cfg, train_dataset, train_eval_dataset, val_dataset, test_dataset
    )

    return evaluate_and_save(
        model,
        cfg,
        {"val": loaders["val"], "test": loaders["test"]},
        train_seconds=0.0,
        checkpoint=args.checkpoint,
        notes="evaluated from an existing checkpoint",
        preserve_training_metrics=True,
    )


def run_predict(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prototypes = None
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)
        cfg.method = model.method
        cfg.model_name = model.model_name
        cfg.num_classes = model.num_classes
    elif args.prototypes:
        payload = np.load(args.prototypes, allow_pickle=False)
        prototypes = payload["prototypes"]
        if "model_name" in payload.files:
            cfg.model_name = str(payload["model_name"].item())
        if "num_classes" in payload.files:
            cfg.num_classes = int(payload["num_classes"].item())
        cfg.method = "ncm"
        model = DINOv3Classifier(
            model_name=cfg.model_name,
            num_classes=cfg.num_classes,
            method="ncm",
            metric_fns=None,
        )
    else:
        raise ValueError("Provide --checkpoint or --prototypes for inference")

    transform = build_transforms(cfg.model_name, augment=False)
    paths = collect_images(args.images, args.image_dir)
    rows = predict_paths(
        model,
        paths,
        transform,
        device,
        top_k=args.top_k,
        prototypes=prototypes,
        batch_size=args.batch_size,
    )

    for row in rows:
        top = ", ".join(
            f"{item['class_id']}:{item['score']:.3f}" for item in row["top_k"]
        )
        print(
            f"{row['path']} -> class {row['predicted_class']} "
            f"({row['confidence']:.3f}) [{top}]"
        )
    if args.output:
        print(f"[predict] wrote {save_predictions(rows, args.output)}")


def run_sweep(args: argparse.Namespace) -> None:
    seen: set[str] = set()
    methods = []
    for method in args.methods:
        normalized = normalize_method(method)
        if normalized not in seen:
            seen.add(normalized)
            methods.append(normalized)

    for method in methods:
        metrics_path = (
            resolve_path(args.results_dir)
            / "flowers102"
            / method
            / f"seed_{args.seed}"
            / "metrics.json"
        )
        if args.skip_existing and metrics_path.exists():
            print(f"[sweep] skipping {method} (found {metrics_path})")
            continue

        command = [
            sys.executable,
            "-m",
            "src.dino_v3.main",
            "train",
            "--method",
            method,
            *_common_flags(args),
        ]
        print(f"[sweep] running {' '.join(command)}")
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    build_report(resolve_path(args.results_dir))


def run_report(args: argparse.Namespace) -> None:
    build_report(resolve_path(args.results_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Few-shot Flowers102 adaptation of a DINOv3 foundation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train one adaptation strategy and evaluate it"
    )
    add_common_arguments(train_parser)
    train_parser.set_defaults(func=run_train_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate an existing checkpoint"
    )
    add_common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.set_defaults(func=run_evaluate)

    predict_parser = subparsers.add_parser(
        "predict", help="Run inference on images with a checkpoint or NCM prototypes"
    )
    predict_parser.add_argument("--model-name", default="vit_base_patch16_dinov3.lvd1689m")
    predict_parser.add_argument("--checkpoint")
    predict_parser.add_argument("--prototypes", help="prototypes.npz from an NCM run")
    predict_parser.add_argument("--images", nargs="+", help="Image file paths")
    predict_parser.add_argument("--image-dir", help="Directory of images")
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument("--batch-size", type=int, default=16)
    predict_parser.add_argument("--output", help="Write JSON or CSV predictions here")
    predict_parser.set_defaults(func=run_predict)

    sweep_parser = subparsers.add_parser(
        "sweep", help="Train every strategy, then build the comparison report"
    )
    add_common_arguments(sweep_parser)
    sweep_parser.add_argument("--methods", nargs="+", default=list(METHODS))
    sweep_parser.add_argument("--skip-existing", action="store_true")
    sweep_parser.set_defaults(func=run_sweep)

    report_parser = subparsers.add_parser(
        "report", help="Aggregate run metrics into tables and plots"
    )
    report_parser.add_argument("--results-dir", default="results/dino_v3")
    report_parser.set_defaults(func=run_report)

    return parser


def run_train_command(args: argparse.Namespace) -> None:
    run_train(config_from_args(args))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
