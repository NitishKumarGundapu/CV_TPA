from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS = ("full", "lp", "lora", "bitfit", "ncm")
METHOD_ALIASES = {"peft": "lora", "zs": "ncm", "linear": "lp"}

DATASETS = ("flowers102", "cifar100")
DATASET_ALIASES = {"cifar-100": "cifar100", "cifar_100": "cifar100", "flowers": "flowers102"}
DATASET_CLASSES = {"flowers102": 102, "cifar100": 100}


def normalize_dataset(name: str) -> str:
    dataset = name.strip().lower()
    dataset = DATASET_ALIASES.get(dataset, dataset)
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose from {DATASETS} "
            f"(aliases: {tuple(DATASET_ALIASES)})"
        )
    return dataset


def normalize_method(name: str) -> str:
    method = name.strip().lower()
    method = METHOD_ALIASES.get(method, method)
    if method not in METHODS:
        raise ValueError(
            f"Unknown method {name!r}. Choose from {METHODS} "
            f"(aliases: {tuple(METHOD_ALIASES)})"
        )
    return method


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class Config:
    method: str = "lp"
    model_name: str = "vit_base_patch16_dinov3.lvd1689m"
    dataset: str = "flowers102"
    data_root: str = "data"
    num_classes: int = 102
    shots_per_class: int | None = 10
    val_per_class: int = 10
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    max_epochs: int = 30
    patience: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.05
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    log_dir: str = "src/dino_v3/logs"
    results_dir: str = "results/dino_v3"
    tsne_samples: int = 1500
    grad_cam_samples: int = 8
    max_eval_batches: int = 0
    limit_batches: float = 1.0

    def __post_init__(self) -> None:
        self.method = normalize_method(self.method)
        self.dataset = normalize_dataset(self.dataset)
        self.num_classes = DATASET_CLASSES[self.dataset]
        if self.shots_per_class is not None and self.shots_per_class < 1:
            raise ValueError("shots_per_class must be a positive integer")
        if self.val_per_class < 1:
            raise ValueError("val_per_class must be a positive integer")
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2")

    @property
    def run_dir(self) -> Path:
        return resolve_path(self.results_dir) / self.dataset / self.method / f"seed_{self.seed}"


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", default="lp",
                        help="Adaptation strategy: full | lp | lora | bitfit | ncm "
                             "(aliases: peft -> lora, zs -> ncm, linear -> lp)")
    parser.add_argument("--model-name", default="vit_base_patch16_dinov3.lvd1689m",
                        help="timm DINOv3 model name")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="flowers102",
                        help="Dataset: flowers102 | cifar100 (aliases: cifar-100)")
    parser.add_argument("--shots-per-class", type=int, default=10,
                        help="Training examples per class; 0 uses the full training split")
    parser.add_argument("--val-per-class", type=int, default=10,
                        help="CIFAR-100 validation images carved out per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--log-dir", default="src/dino_v3/logs")
    parser.add_argument("--results-dir", default="results/dino_v3")
    parser.add_argument("--tsne-samples", type=int, default=1500)
    parser.add_argument("--grad-cam-samples", type=int, default=8)
    parser.add_argument("--max-eval-batches", type=int, default=0,
                        help="Cap evaluation batches per split (0 = all)")
    parser.add_argument("--limit-batches", type=float, default=1.0,
                        help="Lightning limit_train/val_batches, useful for smoke tests")


def config_from_args(args: argparse.Namespace) -> Config:
    values = dict(vars(args))
    if "epochs" in values:
        values.setdefault("max_epochs", values["epochs"])
    kwargs = {}
    for field in fields(Config):
        if field.name in values:
            kwargs[field.name] = values[field.name]
    if kwargs.get("shots_per_class") == 0:
        kwargs["shots_per_class"] = None
    return Config(**kwargs)
