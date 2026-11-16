from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix as _sklearn_confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == targets).float().mean()


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> torch.Tensor:
    k = min(k, logits.shape[-1])
    topk = logits.topk(k, dim=-1).indices
    return (topk == targets.unsqueeze(-1)).any(dim=-1).float().mean()


def classification_metrics(
    logits: np.ndarray, labels: np.ndarray, num_classes: int
) -> dict[str, float]:
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds = logits.argmax(axis=1)
    k = min(5, logits.shape[1])
    return {
        "accuracy": float((preds == labels).mean()),
        "top5_accuracy": float(
            top_k_accuracy_score(labels, logits, k=k, labels=np.arange(num_classes))
        ),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def per_class_f1_rows(
    labels: np.ndarray, preds: np.ndarray, num_classes: int
) -> list[dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=np.arange(num_classes),
        zero_division=0,
    )
    return [
        {
            "class_id": class_id,
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id in range(num_classes)
    ]


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    return _sklearn_confusion_matrix(labels, preds, labels=np.arange(num_classes))
