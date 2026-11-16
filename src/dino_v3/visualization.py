from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    title: str,
    num_classes: int,
    max_samples: int = 1500,
    seed: int = 42,
) -> Path | None:
    if len(labels) < 10:
        return None

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(labels), size=min(max_samples, len(labels)), replace=False)
    points = embeddings[indices]
    perplexity = min(30.0, max(5.0, (len(indices) - 1) / 3.0))
    projected = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(points)

    figure, axis = plt.subplots(figsize=(8, 7))
    scatter = axis.scatter(
        projected[:, 0],
        projected[:, 1],
        c=labels[indices],
        cmap="tab20",
        s=8,
        vmin=-0.5,
        vmax=num_classes - 0.5,
    )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(scatter, ax=axis, label="class ID")
    figure.tight_layout()
    out_path = Path(out_path)
    figure.savefig(out_path, dpi=160)
    plt.close(figure)
    return out_path


def plot_confusion_matrix(
    matrix: np.ndarray,
    out_path: str | Path,
    title: str,
) -> Path:
    normalized = matrix.astype(np.float64)
    normalized = normalized / np.clip(normalized.sum(axis=1, keepdims=True), 1, None)

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xlabel("predicted class")
    axis.set_ylabel("true class")
    axis.set_title(title)
    ticks = list(range(0, normalized.shape[0], 10))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.tick_params(axis="both", labelsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, label="row-normalized rate")
    figure.tight_layout()
    out_path = Path(out_path)
    figure.savefig(out_path, dpi=160)
    plt.close(figure)
    return out_path


def _denormalize(
    image: torch.Tensor,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
) -> np.ndarray:
    mean = torch.tensor(image_mean, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=image.dtype).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _last_block(backbone) -> torch.nn.Module:
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise AttributeError("Could not locate the DINOv3 transformer blocks")
    return blocks[-1]


def _gradcam_for_image(
    model,
    pixel_values: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor | None, int]:
    backbone = model.get_backbone()
    target_layer = _last_block(backbone)
    prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 0))
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(module, inputs, output):
        captured["activation"] = output[0] if isinstance(output, (tuple, list)) else output

    def backward_hook(module, grad_input, grad_output):
        captured["gradient"] = (
            grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        )

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        with torch.enable_grad():
            inputs = pixel_values.to(device).detach().clone().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(inputs)
            prediction = int(logits.argmax(dim=-1)[0])
            logits[0, prediction].backward()

        activation = captured.get("activation")
        gradient = captured.get("gradient")
        if activation is None or gradient is None:
            return None, prediction

        patch_activation = activation[0, prefix_tokens:].detach()
        patch_gradient = gradient[0, prefix_tokens:].detach()
        weights = patch_gradient.mean(dim=0)
        cam = torch.relu((weights.unsqueeze(0) * patch_activation).sum(dim=-1))

        side = int(round(math.sqrt(cam.numel())))
        if side * side != cam.numel():
            return None, prediction

        cam = cam.reshape(1, 1, side, side)
        cam = F.interpolate(
            cam,
            size=tuple(pixel_values.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu(), prediction
    finally:
        forward_handle.remove()
        backward_handle.remove()


def plot_gradcam_grid(
    model,
    dataloader,
    out_path: str | Path,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    num_samples: int = 8,
    device: torch.device | None = None,
) -> Path | None:
    if num_samples < 1:
        return None

    device = device or next(model.parameters()).device
    model.eval()

    candidates: dict[int, torch.Tensor] = {}
    for batch in dataloader:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        for index in range(pixel_values.shape[0]):
            label = int(labels[index])
            if label not in candidates:
                candidates[label] = pixel_values[index].clone()

    if not candidates:
        return None

    class_ids = sorted(candidates)
    if len(class_ids) > num_samples:
        positions = np.linspace(0, len(class_ids) - 1, num_samples)
        class_ids = [
            class_ids[position] for position in np.unique(positions.round().astype(int))
        ]

    samples: list[tuple[torch.Tensor, int, int, torch.Tensor | None]] = []
    for class_id in class_ids:
        image = candidates[class_id]
        cam, prediction = _gradcam_for_image(model, image.unsqueeze(0), device)
        samples.append((image, class_id, prediction, cam))

    columns = min(4, len(samples))
    rows = int(math.ceil(len(samples) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, (image, target, prediction, cam) in zip(axes, samples):
        axis.imshow(_denormalize(image, image_mean, image_std))
        if cam is not None:
            axis.imshow(cam.numpy(), cmap="jet", alpha=0.45)
        axis.set_title(f"true={target} pred={prediction}", fontsize=9)
        axis.axis("off")

    for axis in axes[len(samples) :]:
        axis.axis("off")

    figure.suptitle("Grad-CAM (last DINOv3 block)")
    figure.tight_layout()
    out_path = Path(out_path)
    figure.savefig(out_path, dpi=160)
    plt.close(figure)
    return out_path
