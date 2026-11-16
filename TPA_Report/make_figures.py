from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["full", "lp", "lora", "bitfit", "ncm"]
METHOD_LABELS = {
    "full": "Full",
    "lp": "Linear probe",
    "lora": "LoRA",
    "bitfit": "BitFit",
    "ncm": "NCM",
}
MODEL_COLORS = {"ViT-B/16": "#3B6FB6", "DINOv3-B/16": "#D9822B"}
MODEL_MARKERS = {"ViT-B/16": "o", "DINOv3-B/16": "s"}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 200,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)


def read_summary(path: Path, model: str) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("seed") not in (None, "42"):
                continue
            for key in (
                "accuracy",
                "top5_accuracy",
                "macro_f1",
                "weighted_f1",
                "trainable_fraction",
                "training_seconds",
                "peak_gpu_memory_mb",
            ):
                row[key] = float(row[key]) if row.get(key) not in (None, "") else 0.0
            row["trainable_parameters"] = int(float(row["trainable_parameters"]))
            row["model_label"] = model
            rows.append(row)
    return rows


def all_rows() -> list[dict]:
    rows = read_summary(ROOT / "results" / "analysis" / "summary.csv", "ViT-B/16")
    rows += read_summary(
        ROOT / "results" / "dino_v3" / "analysis" / "summary.csv", "DINOv3-B/16"
    )
    return rows


def grouped_bars(rows: list[dict], dataset: str, out_path: Path) -> None:
    if not rows:
        return
    models = ["ViT-B/16", "DINOv3-B/16"]
    x = np.arange(len(METHODS))
    width = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "accuracy", "Top-1 accuracy"),
        (axes[1], "macro_f1", "Macro F1"),
    ):
        for offset, model in zip((-width / 2, width / 2), models):
            values = []
            for method in METHODS:
                match = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["method"] == method
                    and row["model_label"] == model
                ]
                values.append(match[0][metric] if match else np.nan)
            bars = axis.bar(
                x + offset,
                values,
                width,
                label=model,
                color=MODEL_COLORS[model],
                edgecolor="black",
                linewidth=0.4,
            )
            for bar, value in zip(bars, values):
                if np.isnan(value):
                    continue
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.012,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.6,
                    rotation=0,
                )
        axis.set_xticks(x, [METHOD_LABELS[method] for method in METHODS], rotation=20)
        axis.set_ylim(0, 1.08)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(f"{dataset}: {title}")
    axes[0].legend(frameon=False, loc="lower left")
    figure.savefig(out_path)
    plt.close(figure)


def efficiency(rows: list[dict], dataset: str, out_path: Path) -> None:
    data = [row for row in rows if row["dataset"] == dataset]
    if not data:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for model in ("ViT-B/16", "DINOv3-B/16"):
        model_rows = [row for row in data if row["model_label"] == model]
        for row in model_rows:
            x_params = max(row["trainable_parameters"], 100)
            for axis, x_value in (
                (axes[0], x_params),
                (axes[1], row["training_seconds"]),
            ):
                axis.scatter(
                    x_value,
                    row["accuracy"],
                    s=70,
                    marker=MODEL_MARKERS[model],
                    color=MODEL_COLORS[model],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
                axis.annotate(
                    METHOD_LABELS[row["method"]],
                    (x_value, row["accuracy"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Trainable parameters (log scale; NCM plotted at 10^2)")
    axes[0].set_ylabel("Test top-1 accuracy")
    axes[0].set_title(f"{dataset}: accuracy vs. trainable parameters")
    axes[1].set_xlabel("Training time (s)")
    axes[1].set_title(f"{dataset}: accuracy vs. training time")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_ylim(
            max(0.0, min(row["accuracy"] for row in data) - 0.05),
            min(1.0, max(row["accuracy"] for row in data) + 0.06),
        )
    handles = [
        plt.Line2D(
            [],
            [],
            marker=MODEL_MARKERS[model],
            color="none",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="black",
            markersize=7,
            label=model,
        )
        for model in ("ViT-B/16", "DINOv3-B/16")
    ]
    axes[1].legend(handles=handles, frameon=False, loc="lower right")
    figure.savefig(out_path)
    plt.close(figure)


def per_class_f1(out_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    for axis, model, base in (
        (axes[0], "ViT-B/16", ROOT / "results" / "cifar100"),
        (axes[1], "DINOv3-B/16", ROOT / "results" / "dino_v3" / "cifar100"),
    ):
        series = []
        for method in METHODS:
            path = base / method / "seed_42" / "per_class_f1.csv"
            if not path.exists():
                series.append([])
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                series.append([float(row["f1"]) for row in csv.DictReader(handle)])
        if not any(series):
            continue
        positions = np.arange(1, len(METHODS) + 1)
        boxes = axis.boxplot(
            series,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1},
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.75)
        axis.set_xticks(positions, [METHOD_LABELS[m] for m in METHODS], rotation=20)
        axis.set_title(f"{model}: per-class F1 (CIFAR-100)")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, 1.02)
    axes[0].set_ylabel("Per-class F1")
    figure.savefig(out_path)
    plt.close(figure)


def pipeline(out_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 2.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    steps = [
        ("Data", "Flowers102\nCIFAR-100"),
        ("Splits", "10-shot train\nheld-out val/test"),
        ("Backbone", "ViT-B/16\nDINOv3-B/16"),
        ("Adaptation", "Full / LP\nLoRA / BitFit / NCM"),
        ("Evaluation", "Acc, F1, params\ntime, memory"),
        ("Artifacts", "t-SNE, Grad-CAM\nreport, PDF"),
    ]
    width = 0.145
    gap = 0.018
    total = len(steps) * width + (len(steps) - 1) * gap
    start = (1 - total) / 2
    colors = ["#E8EEF7", "#E8EEF7", "#E8EEF7", "#E8EEF7", "#E8EEF7", "#E8EEF7"]
    for index, ((title, subtitle), color) in enumerate(zip(steps, colors)):
        x = start + index * (width + gap)
        box = FancyBboxPatch(
            (x, 0.25),
            width,
            0.5,
            boxstyle="round,pad=0.006,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#3B4A5A",
            facecolor=color,
        )
        axis.add_patch(box)
        axis.text(
            x + width / 2,
            0.62,
            title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        axis.text(
            x + width / 2,
            0.41,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.5,
        )
        if index < len(steps) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (x + width + 0.002, 0.5),
                    (x + width + gap - 0.002, 0.5),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.8,
                    color="#3B4A5A",
                )
            )
    figure.savefig(out_path)
    plt.close(figure)


def comparison_grid(
    images: list[Path],
    titles: list[str],
    out_path: Path,
    figsize: tuple[float, float],
    columns: int = 1,
    height_ratios: list[float] | None = None,
) -> None:
    rows = int(np.ceil(len(images) / columns))
    grid = plt.GridSpec(
        rows,
        columns,
        figure=plt.figure(figsize=figsize),
        height_ratios=height_ratios,
        wspace=0.04,
        hspace=0.08,
    )
    figure = plt.gcf()
    for index, (path, title) in enumerate(zip(images, titles)):
        axis = figure.add_subplot(grid[index // columns, index % columns])
        axis.imshow(mpimg.imread(path))
        axis.axis("off")
        axis.set_title(title, fontsize=9)
    figure.savefig(out_path)
    plt.close(figure)


def main() -> None:
    rows = all_rows()
    cifar = [row for row in rows if row["dataset"] == "cifar100"]
    flowers = [row for row in rows if row["dataset"] == "flowers102"]

    grouped_bars(cifar, "cifar100", OUT / "fig_cifar_bars.png")
    grouped_bars(flowers, "flowers102", OUT / "fig_flowers_bars.png")
    efficiency(cifar, "cifar100", OUT / "fig_cifar_efficiency.png")
    per_class_f1(OUT / "fig_per_class_f1.png")
    pipeline(OUT / "fig_pipeline.png")

    tSNE_paths = [
        ROOT / "results" / "cifar100" / "full" / "seed_42" / "figures" / "tsne_val.png",
        ROOT / "results" / "dino_v3" / "cifar100" / "lp" / "seed_42" / "figures" / "tsne_val.png",
    ]
    comparison_grid(
        tSNE_paths,
        ["(a) ViT-B/16 full fine-tuning", "(b) DINOv3-B/16 linear probe"],
        OUT / "fig_tsne_compare.png",
        (12, 5.0),
        columns=2,
    )

    gradcam_paths = [
        ROOT / "results" / "cifar100" / "full" / "seed_42" / "figures" / "gradcam_val.png",
        ROOT / "results" / "dino_v3" / "cifar100" / "full" / "seed_42" / "figures" / "gradcam_val.png",
    ]
    comparison_grid(
        gradcam_paths,
        ["(a) ViT-B/16 full fine-tuning", "(b) DINOv3-B/16 full fine-tuning"],
        OUT / "fig_gradcam_compare.png",
        (11, 7.2),
        columns=1,
        height_ratios=[1.0, 2.0],
    )

    confusion_paths = [
        ROOT / "results" / "cifar100" / "full" / "seed_42" / "figures" / "confusion_test.png",
        ROOT / "results" / "dino_v3" / "cifar100" / "lp" / "seed_42" / "figures" / "confusion_test.png",
    ]
    comparison_grid(
        confusion_paths,
        ["(a) ViT-B/16 full fine-tuning", "(b) DINOv3-B/16 linear probe"],
        OUT / "fig_confusion_compare.png",
        (12, 4.6),
        columns=2,
    )
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
