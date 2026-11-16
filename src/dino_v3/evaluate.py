from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import Config
from .dataset import resolve_model_data_config
from .metrics import classification_metrics, confusion_matrix, per_class_f1_rows
from .visualization import plot_confusion_matrix, plot_gradcam_grid, plot_tsne

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@torch.inference_mode()
def extract_outputs(
    model,
    dataloader,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, np.ndarray]:
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []

    for index, batch in enumerate(dataloader):
        if max_batches and index >= max_batches:
            break
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        split_logits, features = model.encode_with_logits(pixel_values)
        logits.append(split_logits.float().cpu())
        embeddings.append(features.float().cpu())
        labels.append(batch["labels"].cpu())

    if not logits:
        raise RuntimeError("No batches were extracted; check max_eval_batches")

    return {
        "logits": torch.cat(logits).numpy(),
        "labels": torch.cat(labels).numpy(),
        "embeddings": torch.cat(embeddings).numpy(),
    }


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-8)


def nearest_class_mean(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    features = _l2_normalize(embeddings)
    prototypes = np.zeros((num_classes, features.shape[1]), dtype=np.float32)
    for class_id in range(num_classes):
        mask = labels == class_id
        if mask.any():
            prototypes[class_id] = features[mask].mean(axis=0)
    return _l2_normalize(prototypes)


def predict_prototypes(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    scale: float = 20.0,
) -> np.ndarray:
    similarities = _l2_normalize(embeddings) @ prototypes.T
    return (scale * similarities).astype(np.float32)


def load_checkpoint(path: str | Path):
    from .model import DINOv3Classifier

    return DINOv3Classifier.load_from_checkpoint(
        str(path), map_location="cpu", metric_fns=None
    )


def collect_images(images: list[str] | None, image_dir: str | None) -> list[Path]:
    paths: list[Path] = []
    if images:
        paths.extend(Path(image).expanduser() for image in images)
    if image_dir:
        directory = Path(image_dir).expanduser()
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        paths.extend(
            path
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not paths:
        raise ValueError("Provide at least one image with --images or --image-dir")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Image(s) not found: {', '.join(missing)}")
    return paths


@torch.inference_mode()
def predict_paths(
    model,
    paths: list[Path],
    transform,
    device: torch.device,
    top_k: int = 5,
    prototypes: np.ndarray | None = None,
    batch_size: int = 16,
) -> list[dict]:
    model.to(device).eval()
    batch_size = max(1, batch_size)
    top_k = max(1, min(top_k, model.num_classes))

    results = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        pixel_values = torch.stack(
            [transform(Image.open(path).convert("RGB")) for path in batch_paths]
        ).to(device)
        logits, features = model.encode_with_logits(pixel_values)
        logits_np = logits.float().cpu().numpy()
        if prototypes is not None:
            logits_np = predict_prototypes(features.float().cpu().numpy(), prototypes)

        scores = torch.from_numpy(logits_np).softmax(dim=-1).numpy()
        k = min(top_k, scores.shape[1])
        for path, row in zip(batch_paths, scores):
            order = np.argsort(row)[::-1][:k]
            results.append(
                {
                    "path": str(path),
                    "predicted_class": int(order[0]),
                    "confidence": float(row[order[0]]),
                    "top_k": [
                        {"class_id": int(class_id), "score": float(row[class_id])}
                        for class_id in order
                    ],
                }
            )
    return results


def save_predictions(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["path", "predicted_class", "confidence", "top_k"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "top_k": json.dumps(row["top_k"]),
                    }
                )
    else:
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_and_save(
    model,
    cfg: Config,
    loaders: dict,
    *,
    train_seconds: float,
    peak_gpu_memory_mb: float | None = None,
    best_val_loss: float | None = None,
    checkpoint: str | Path | None = None,
    prototypes: np.ndarray | None = None,
    image_mean: tuple[float, ...] | None = None,
    image_std: tuple[float, ...] | None = None,
    notes: str = "",
    preserve_training_metrics: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    output_dir = cfg.run_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, np.ndarray]] = {}
    for split in ("val", "test"):
        split_outputs = extract_outputs(model, loaders[split], device, cfg.max_eval_batches)
        if prototypes is not None:
            split_outputs["logits"] = predict_prototypes(
                split_outputs["embeddings"], prototypes
            )
        outputs[split] = split_outputs

    np.savez_compressed(
        output_dir / "predictions.npz",
        **{
            f"{split}_{name}": value
            for split, split_outputs in outputs.items()
            for name, value in split_outputs.items()
        },
    )

    results = {
        split: classification_metrics(
            split_outputs["logits"], split_outputs["labels"], cfg.num_classes
        )
        for split, split_outputs in outputs.items()
    }

    test_labels = outputs["test"]["labels"]
    test_preds = outputs["test"]["logits"].argmax(axis=1)
    _write_csv(
        output_dir / "per_class_f1.csv",
        per_class_f1_rows(test_labels, test_preds, cfg.num_classes),
        ["class_id", "precision", "recall", "f1", "support"],
    )
    matrix = confusion_matrix(test_labels, test_preds, cfg.num_classes)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    plot_confusion_matrix(
        matrix,
        figures_dir / "confusion_test.png",
        title=f"{cfg.dataset} / {cfg.method} / test",
    )

    plot_tsne(
        outputs["val"]["embeddings"],
        outputs["val"]["labels"],
        figures_dir / "tsne_val.png",
        title=f"{cfg.dataset} / {cfg.method} / val embeddings",
        num_classes=cfg.num_classes,
        max_samples=cfg.tsne_samples,
        seed=cfg.seed,
    )

    if prototypes is None:
        data_config = resolve_model_data_config(cfg.model_name)
        image_mean = image_mean or tuple(float(v) for v in data_config["mean"])
        image_std = image_std or tuple(float(v) for v in data_config["std"])
        plot_gradcam_grid(
            model,
            loaders["val"],
            figures_dir / "gradcam_val.png",
            image_mean,
            image_std,
            num_samples=cfg.grad_cam_samples,
            device=device,
        )

    counts = model.parameter_counts()
    record = {
        "dataset": cfg.dataset,
        "model_family": "dinov3",
        "method": cfg.method,
        "model": cfg.model_name,
        "seed": cfg.seed,
        "shots_per_class": cfg.shots_per_class,
        "epochs": 0 if prototypes is not None else cfg.max_epochs,
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "trainable_parameters": counts["trainable_parameters"],
        "total_parameters": counts["total_parameters"],
        "trainable_fraction": counts["trainable_fraction"],
        "training_seconds": round(float(train_seconds), 3),
        "peak_gpu_memory_mb": (
            round(float(peak_gpu_memory_mb), 1) if peak_gpu_memory_mb is not None else None
        ),
        "best_val_loss": best_val_loss,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "accuracy": results["test"]["accuracy"],
        "top5_accuracy": results["test"]["top5_accuracy"],
        "macro_f1": results["test"]["macro_f1"],
        "weighted_f1": results["test"]["weighted_f1"],
        "val_accuracy": results["val"]["accuracy"],
        "val_macro_f1": results["val"]["macro_f1"],
        "notes": notes,
    }
    metrics_path = output_dir / "metrics.json"
    if preserve_training_metrics and metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in (
            "epochs",
            "shots_per_class",
            "training_seconds",
            "peak_gpu_memory_mb",
            "best_val_loss",
            "checkpoint",
            "notes",
        ):
            if previous.get(key) is not None:
                record[key] = previous[key]

    metrics_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        f"[{cfg.method}] test acc={record['accuracy']:.4f} "
        f"top5={record['top5_accuracy']:.4f} macro_f1={record['macro_f1']:.4f} "
        f"trainable={record['trainable_parameters']:,} "
        f"time={record['training_seconds']:.1f}s"
    )
    return record
