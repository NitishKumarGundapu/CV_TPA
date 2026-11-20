#!/usr/bin/env python3
"""Generate all figures and LaTeX tables for the TPA report.

Uses the same result artefacts as the presentation.  Every number in the report
tables is written from ``results/*/*/*/seed_*/metrics.json`` and the per-class
CSV files, so the text and tables cannot drift apart.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(__file__).resolve().parent
FIG = REPORT / "figures"
TAB = REPORT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "TPA_Presentation"))
import make_deck_figures as deck  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

DATASETS = ["flowers102", "oxfordiiitpet"]
FAMILIES = ["vit", "dino_v3"]
METHODS = ["full", "lp", "lora", "bitfit", "ncm"]
TRAINED = ["full", "lp", "lora", "bitfit"]
PRETTY_DS = {"flowers102": "Flowers102", "oxfordiiitpet": "Oxford-IIIT Pet"}
SHORT_DS = {"flowers102": "Flowers", "oxfordiiitpet": "Pet"}
PRETTY_FAM = {"vit": "ViT-B/16", "dino_v3": "DINOv3-B/16"}
PRETTY_METHOD = {
    "full": "Full",
    "lp": "Linear probe",
    "lora": "LoRA",
    "bitfit": "BitFit",
    "ncm": "NCM",
}
SHORT_METHOD = {"full": "Full", "lp": "LP", "lora": "LoRA", "bitfit": "BitFit", "ncm": "NCM"}
MODEL_COLOR = {"vit": "#4C72B0", "dino_v3": "#C44E52"}


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _runs() -> dict:
    return deck.load_runs()


def _pct(value: float) -> float:
    return 100.0 * value


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _best_bold(values: list[float], text: list[str]) -> list[str]:
    best = max(values)
    return [f"\\textbf{{{item}}}" if abs(value - best) < 1e-12 else item for value, item in zip(values, text)]


# --------------------------------------------------------------------------- #
# Schematic figures
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, facecolor="#EAF0F8", edgecolor="#355070"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2)


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color="#355070",
            shrinkA=2,
            shrinkB=2,
        )
    )


def pipeline_figure() -> None:
    figure, axis = plt.subplots(figsize=(7.2, 2.15))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 2.6)
    axis.axis("off")
    _box(axis, 0.15, 0.75, 1.55, 1.1, "Dataset\nFlowers102 /\nOxford-IIIT Pet")
    _box(axis, 2.05, 0.75, 1.65, 1.1, "Preprocessing\nresize, normalise,\naugment")
    _box(axis, 4.05, 0.75, 1.75, 1.1, "Backbone\nViT-B/16 or\nDINOv3-B/16")
    _box(axis, 6.15, 0.75, 1.75, 1.1, "Adaptation\nFull / LP / LoRA /\nBitFit / NCM")
    _box(axis, 8.15, 0.75, 1.70, 1.1, "Evaluation\nmetrics, curves,\nconfusion,\nt-SNE, Grad-CAM")
    for start, end in [
        ((1.70, 1.30), (2.05, 1.30)),
        ((3.70, 1.30), (4.05, 1.30)),
        ((5.80, 1.30), (6.15, 1.30)),
        ((7.90, 1.30), (8.15, 1.30)),
    ]:
        _arrow(axis, start, end)
    figure.savefig(FIG / "fig_pipeline.png")
    plt.close(figure)
    print("wrote fig_pipeline.png")


def methods_figure() -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.1))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6.2)
    axis.axis("off")

    frozen, trainable, adapter, bias, prototype = (
        "#D9D9D9",
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#CCB974",
    )
    y_positions = [5.0, 4.0, 3.0, 2.0, 1.0]
    for y, method in zip(y_positions, METHODS):
        axis.text(0.05, y + 0.28, PRETTY_METHOD[method], fontsize=8, ha="left", va="center")
        axis.add_patch(Rectangle((1.65, y), 3.6, 0.56, facecolor=frozen, edgecolor="#666666", linewidth=0.7))
        axis.text(3.45, y + 0.28, "Backbone", ha="center", va="center", fontsize=7)
        axis.add_patch(Rectangle((5.55, y), 1.1, 0.56, facecolor=frozen, edgecolor="#666666", linewidth=0.7))
        axis.text(6.10, y + 0.28, "Head", ha="center", va="center", fontsize=7)

    axis.patches[0].set_facecolor(trainable)  # Full: backbone
    axis.patches[1].set_facecolor(trainable)  # Full: head
    axis.patches[3].set_facecolor(trainable)  # LP: head

    for x in (1.90, 2.85, 3.80):
        axis.add_patch(Rectangle((x, 3.0), 0.55, 0.12, facecolor=adapter, edgecolor="#8C4A2F", linewidth=0.6))
    axis.patches[5].set_facecolor(trainable)  # LoRA: head

    for x in (1.80, 2.55, 3.30, 4.05, 4.80):
        axis.add_patch(Rectangle((x, 2.56), 0.50, 0.10, facecolor=bias, edgecolor="#3E6B4F", linewidth=0.4))
    axis.patches[7].set_facecolor(trainable)  # BitFit: head

    axis.add_patch(Rectangle((7.15, 1.0), 1.55, 0.56, facecolor=prototype, edgecolor="#8C7A2F", linewidth=0.7))
    axis.text(7.92, 1.28, "Class prototypes", ha="center", va="center", fontsize=7)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=trainable, edgecolor="#666666", linewidth=0.6),
        Rectangle((0, 0), 1, 1, facecolor=frozen, edgecolor="#666666", linewidth=0.6),
        Rectangle((0, 0), 1, 1, facecolor=adapter, edgecolor="#8C4A2F", linewidth=0.6),
        Rectangle((0, 0), 1, 1, facecolor=bias, edgecolor="#3E6B4F", linewidth=0.6),
        Rectangle((0, 0), 1, 1, facecolor=prototype, edgecolor="#8C7A2F", linewidth=0.6),
    ]
    axis.legend(
        handles,
        ["Trainable", "Frozen", "LoRA adapter", "Bias term", "Training-free"],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.50, 0.00),
    )
    figure.savefig(FIG / "fig_methods.png")
    plt.close(figure)
    print("wrote fig_methods.png")


# --------------------------------------------------------------------------- #
# Dataset samples and architecture images
# --------------------------------------------------------------------------- #
def _sample_grid(samples, out: Path, title: str) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(6.8, 4.2), gridspec_kw={"hspace": 0.32, "wspace": 0.04})
    for axis, (image, label) in zip(axes.ravel(), samples):
        axis.imshow(image)
        axis.set_title(label, fontsize=8, pad=4)
        axis.axis("off")
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    print("wrote", out.name)


def dataset_samples() -> None:
    labels = sio.loadmat(ROOT / "data/flowers-102/imagelabels.mat")["labels"][0]
    split = sio.loadmat(ROOT / "data/flowers-102/setid.mat")["tstid"].ravel()
    chosen = [
        (1, "pink primrose"),
        (2, "hard-leaved pocket orchid"),
        (6, "tiger lily"),
        (13, "king protea"),
        (54, "sunflower"),
        (74, "rose"),
    ]
    samples = []
    for class_id, name in chosen:
        ids = [int(i) for i in split if int(labels[int(i) - 1]) == class_id]
        image_id = ids[len(ids) // 2]
        image = Image.open(ROOT / f"data/flowers-102/jpg/image_{image_id:05d}.jpg").convert("RGB")
        image = Image.fromarray(np.asarray(image.resize((420, 320))))
        samples.append((image, f"{name}\n(class {class_id})"))
    _sample_grid(samples, FIG / "fig_flowers_samples.png", "Flowers102: six of the 102 classes")

    chosen = ["beagle", "persian", "pug", "siamese", "boxer", "maine_coon"]
    samples = []
    for name in chosen:
        files = sorted((ROOT / "data/oxford_pet/test" / name).glob("*.jpg"))
        image = Image.open(files[len(files) // 2]).convert("RGB")
        image = Image.fromarray(np.asarray(image.resize((420, 320))))
        samples.append((image, name.replace("_", " ")))
    _sample_grid(samples, FIG / "fig_pet_samples.png", "Oxford-IIIT Pet: six of the 37 breeds")


def architecture_images() -> None:
    Image.open(ROOT / "archs" / "vit_arch.png").convert("RGB").save(FIG / "fig_arch_vit.png")
    image = Image.open(ROOT / "archs" / "dino_v3.webp").convert("RGBA")
    canvas = Image.new("RGB", image.size, "white")
    canvas.paste(image, mask=image.split()[-1])
    canvas.save(FIG / "fig_arch_dino.png")
    print("wrote architecture images")


# --------------------------------------------------------------------------- #
# Results plots
# --------------------------------------------------------------------------- #
def loss_curves(runs: dict) -> None:
    for family in FAMILIES:
        for dataset in DATASETS:
            with plt.rc_context(
                {
                    "font.size": 8,
                    "axes.titlesize": 8.5,
                    "axes.labelsize": 7.5,
                    "legend.fontsize": 6.5,
                    "xtick.labelsize": 6.5,
                    "ytick.labelsize": 6.5,
                }
            ):
                figure, axes = plt.subplots(2, 2, figsize=(7.0, 3.2), sharex=True)
                for axis, method in zip(axes.ravel(), TRAINED):
                    record = runs[(family, dataset, method)]
                    epochs = deck.parse_epochs(Path(record["checkpoint"]).parents[1] / "metrics.csv")
                    keys = sorted(epochs)
                    train = [epochs[e].get("train_loss", np.nan) for e in keys]
                    val = [epochs[e].get("val_loss", np.nan) for e in keys]
                    accuracy = [epochs[e].get("val_acc", np.nan) for e in keys]
                    axis2 = axis.twinx()
                    line1, = axis.plot(keys, train, "o-", color="#4C72B0", markersize=2.4, label="train loss")
                    line2, = axis.plot(keys, val, "s-", color="#C44E52", markersize=2.4, label="val loss")
                    line3, = axis2.plot(keys, accuracy, "--", color="#55A868", linewidth=1.3, label="val acc")
                    axis.set_yscale("log")
                    axis.set_title(f"{PRETTY_METHOD[method]} ({_pct(record['accuracy']):.2f}% test)")
                    axis.set_xlabel("epoch")
                    axis.set_ylabel("loss (log)")
                    axis2.set_ylabel("val acc")
                    if not np.all(np.isnan(accuracy)):
                        axis2.set_ylim(max(0.0, np.nanmin(accuracy) - 0.04), 1.0)
                    axis.grid(alpha=0.25, which="both")
                    axis.legend([line1, line2, line3], ["train", "val", "val acc"], loc="best")
                figure.tight_layout(pad=0.3)
                out = FIG / f"fig_loss_{family}_{dataset}.png"
                figure.savefig(out)
                plt.close(figure)
                print("wrote", out.name)


def accuracy_bars(runs: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    width = 0.38
    x = np.arange(len(METHODS))
    for axis, dataset in zip(axes, DATASETS):
        vit = [_pct(runs[("vit", dataset, m)]["accuracy"]) for m in METHODS]
        dino = [_pct(runs[("dino_v3", dataset, m)]["accuracy"]) for m in METHODS]
        bars1 = axis.bar(x - width / 2, vit, width, color=MODEL_COLOR["vit"], label="ViT-B/16")
        bars2 = axis.bar(x + width / 2, dino, width, color=MODEL_COLOR["dino_v3"], label="DINOv3-B/16")
        for bars in (bars1, bars2):
            for bar in bars:
                axis.annotate(
                    f"{bar.get_height():.1f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points",
                    xytext=(0, 2),
                    ha="center",
                    fontsize=5.8,
                )
        axis.set_xticks(x)
        axis.set_xticklabels([SHORT_METHOD[m] for m in METHODS])
        axis.set_ylabel("test top-1 accuracy (%)")
        axis.set_title(PRETTY_DS[dataset])
        low = min(min(vit), min(dino))
        high = max(max(vit), max(dino))
        axis.set_ylim(low - 1.0, high + 1.2)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="lower right", frameon=True)
    figure.tight_layout()
    figure.savefig(FIG / "fig_accuracy_bars.png")
    plt.close(figure)
    print("wrote fig_accuracy_bars.png")


def efficiency(runs: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.4))
    for row, dataset in enumerate(DATASETS):
        for col, x_key in enumerate(["trainable_parameters", "training_seconds"]):
            axis = axes[row, col]
            for family in FAMILIES:
                for method in METHODS:
                    record = runs[(family, dataset, method)]
                    x_value = record[x_key]
                    if x_key == "trainable_parameters":
                        x_value = max(x_value, 1)
                    axis.scatter(
                        x_value,
                        _pct(record["accuracy"]),
                        s=32,
                        color=MODEL_COLOR[family],
                        marker="o" if family == "vit" else "s",
                        edgecolor="white",
                        linewidth=0.4,
                        zorder=3,
                    )
                    axis.annotate(
                        SHORT_METHOD[method],
                        (x_value, _pct(record["accuracy"])),
                        textcoords="offset points",
                        xytext=(3, 3),
                        fontsize=5.8,
                    )
            if col == 0:
                axis.set_xscale("log")
                axis.set_xlabel("trainable parameters (log scale)")
            else:
                axis.set_xlabel("training time (s)")
            axis.set_ylabel("test top-1 accuracy (%)")
            axis.set_title(PRETTY_DS[dataset])
            axis.grid(alpha=0.25)
            low = min(_pct(runs[(f, dataset, m)]["accuracy"]) for f in FAMILIES for m in METHODS)
            high = max(_pct(runs[(f, dataset, m)]["accuracy"]) for f in FAMILIES for m in METHODS)
            axis.set_ylim(low - 0.5, high + 0.5)
    axes[0, 0].scatter([], [], color=MODEL_COLOR["vit"], marker="o", label="ViT-B/16")
    axes[0, 0].scatter([], [], color=MODEL_COLOR["dino_v3"], marker="s", label="DINOv3-B/16")
    axes[0, 0].legend(loc="lower right", frameon=True)
    figure.tight_layout()
    figure.savefig(FIG / "fig_efficiency.png")
    plt.close(figure)
    print("wrote fig_efficiency.png")


def per_class_f1() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for axis, dataset in zip(axes, DATASETS):
        positions = []
        data = []
        colors = []
        for index, method in enumerate(METHODS):
            for family in FAMILIES:
                path = (
                    ROOT
                    / "results"
                    / family
                    / dataset
                    / method
                    / "seed_42"
                    / "per_class_f1.csv"
                )
                rows = list(csv.DictReader(path.open(encoding="utf-8")))
                data.append([float(row["f1"]) for row in rows])
                positions.append(index + (-0.18 if family == "vit" else 0.18))
                colors.append(MODEL_COLOR[family])
        box = axis.boxplot(
            data,
            positions=positions,
            widths=0.30,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 0.8},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.4)
        axis.set_xticks(range(len(METHODS)))
        axis.set_xticklabels([SHORT_METHOD[m] for m in METHODS])
        axis.set_ylabel("per-class F1")
        axis.set_title(PRETTY_DS[dataset])
        axis.set_ylim(0.55, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].scatter([], [], color=MODEL_COLOR["vit"], marker="s", label="ViT-B/16")
    axes[0].scatter([], [], color=MODEL_COLOR["dino_v3"], marker="s", label="DINOv3-B/16")
    axes[0].legend(loc="lower left", frameon=True)
    figure.tight_layout()
    figure.savefig(FIG / "fig_per_class_f1.png")
    plt.close(figure)
    print("wrote fig_per_class_f1.png")


def confusion(runs: dict) -> None:
    specs = [
        ("vit", "flowers102", "ncm"),
        ("dino_v3", "flowers102", "ncm"),
        ("vit", "oxfordiiitpet", "full"),
        ("dino_v3", "oxfordiiitpet", "ncm"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.4))
    figure.subplots_adjust(left=0.08, right=0.86, bottom=0.13, top=0.90, wspace=0.24, hspace=0.55)
    image = None
    for axis, (family, dataset, method) in zip(axes.ravel(), specs):
        matrix = np.loadtxt(
            ROOT / "results" / family / dataset / method / "seed_42" / "confusion_matrix.csv",
            delimiter=",",
        )
        normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        image = axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(
            f"{PRETTY_FAM[family]} / {SHORT_DS[dataset]} / {SHORT_METHOD[method]} "
            f"({_pct(runs[(family, dataset, method)]['accuracy']):.2f}%)",
            fontsize=8,
        )
        axis.set_xlabel("predicted class")
        axis.set_ylabel("true class")
        axis.set_xticks([])
        axis.set_yticks([])
    colorbar_axis = figure.add_axes([0.89, 0.18, 0.015, 0.64])
    figure.colorbar(image, cax=colorbar_axis, label="row-normalised rate")
    figure.savefig(FIG / "fig_confusion.png")
    plt.close(figure)
    print("wrote fig_confusion.png")


def gradcam() -> None:
    columns = [
        ("flowers102", "full"),
        ("flowers102", "lp"),
        ("oxfordiiitpet", "full"),
        ("oxfordiiitpet", "lp"),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 3.0))
    for row, family in enumerate(FAMILIES):
        for col, (dataset, method) in enumerate(columns):
            if family == "vit":
                path = (
                    ROOT
                    / "results"
                    / family
                    / dataset
                    / method
                    / "seed_42"
                    / "figures"
                    / "gradcam_val.png"
                )
            else:
                path = ROOT / "TPA_Presentation" / "figures" / f"gradcam_dino_{method}_{dataset}.png"
            panel = deck._gradcam_panels(path)[0]
            axis = axes[row, col]
            axis.imshow(panel)
            axis.axis("off")
            if row == 0:
                axis.set_title(f"{SHORT_DS[dataset]} / {SHORT_METHOD[method]}", fontsize=8)
        figure.text(
            0.012,
            0.71 - 0.40 * row,
            PRETTY_FAM[family],
            rotation=90,
            va="center",
            ha="center",
            fontsize=8,
        )
    figure.tight_layout(pad=0.2, rect=(0.03, 0, 1, 1))
    figure.savefig(FIG / "fig_gradcam.png")
    plt.close(figure)
    print("wrote fig_gradcam.png")


def tsne(runs: dict) -> None:
    rng = np.random.default_rng(42)
    for family in FAMILIES:
        figure, axes = plt.subplots(2, 5, figsize=(7.2, 2.8))
        for row, dataset in enumerate(DATASETS):
            for col, method in enumerate(METHODS):
                axis = axes[row, col]
                bundle = np.load(
                    ROOT / "results" / family / dataset / method / "seed_42" / "predictions.npz"
                )
                embeddings = bundle["val_embeddings"]
                labels = bundle["val_labels"]
                indices = rng.choice(len(labels), size=min(600, len(labels)), replace=False)
                perplexity = min(30.0, max(5.0, (len(indices) - 1) / 3.0))
                projected = TSNE(
                    n_components=2,
                    init="pca",
                    learning_rate="auto",
                    perplexity=perplexity,
                    random_state=42,
                    max_iter=750,
                ).fit_transform(embeddings[indices])
                num_classes = 102 if dataset == "flowers102" else 37
                axis.scatter(
                    projected[:, 0],
                    projected[:, 1],
                    c=labels[indices],
                    cmap="turbo",
                    s=1.8,
                    vmin=0,
                    vmax=num_classes - 1,
                    linewidths=0,
                )
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    axis.set_title(PRETTY_METHOD[method], fontsize=7.5)
            axes[row, 0].set_ylabel(
                f"{SHORT_DS[dataset]}\n({102 if dataset == 'flowers102' else 37} classes)",
                fontsize=7.5,
            )
        figure.tight_layout(pad=0.25)
        figure.savefig(FIG / f"fig_tsne_{family}.png")
        plt.close(figure)
        print("wrote", f"fig_tsne_{family}.png")


# --------------------------------------------------------------------------- #
# LaTeX tables
# --------------------------------------------------------------------------- #
def _result_rows(runs: dict, family: str) -> str:
    lines = []
    for dataset in DATASETS:
        records = [runs[(family, dataset, method)] for method in METHODS]
        precision = [deck.aggregate_precision_recall(r)[0] for r in records]
        recall = [deck.aggregate_precision_recall(r)[1] for r in records]
        values = {
            "acc": [_pct(r["accuracy"]) for r in records],
            "top5": [_pct(r["top5_accuracy"]) for r in records],
            "p": [_pct(v) for v in precision],
            "r": [_pct(v) for v in recall],
            "f1": [_pct(r["macro_f1"]) for r in records],
        }
        text = {
            key: [_fmt(v) for v in series] for key, series in values.items()
        }
        for key in ("acc", "top5", "p", "r", "f1"):
            text[key] = _best_bold(values[key], text[key])
        for index, method in enumerate(METHODS):
            record = records[index]
            params = record["trainable_parameters"]
            fraction = _pct(record["trainable_fraction"])
            lines.append(
                " & ".join(
                    [
                        SHORT_METHOD[method],
                        SHORT_DS[dataset],
                        text["acc"][index],
                        text["top5"][index],
                        text["p"][index],
                        text["r"][index],
                        text["f1"][index],
                        f"{params:,}" if params else "0",
                        f"{fraction:.2f}",
                        f"{record['training_seconds']:.1f}",
                        f"{record['peak_gpu_memory_mb']:.0f}",
                    ]
                )
                + r" \\"
            )
        if dataset != DATASETS[-1]:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def _per_run_rows(runs: dict) -> str:
    lines = []
    for family in FAMILIES:
        for dataset in DATASETS:
            for method in METHODS:
                record = runs[(family, dataset, method)]
                checkpoint = record.get("checkpoint")
                epoch = "--"
                best_val_text = "--"
                if checkpoint:
                    best_val = record.get("best_val_loss")
                    if best_val is not None:
                        best_val_text = f"{best_val:.4f}"
                    metrics_csv = Path(checkpoint).parents[1] / "metrics.csv"
                    if metrics_csv.exists():
                        epochs = deck.parse_epochs(metrics_csv)
                        val_epochs = [
                            (values["val_loss"], index)
                            for index, values in epochs.items()
                            if "val_loss" in values
                        ]
                        if val_epochs:
                            epoch = f"{min(val_epochs)[1]:02d}"
                lines.append(
                    " & ".join(
                        [
                            PRETTY_FAM[family],
                            SHORT_DS[dataset],
                            SHORT_METHOD[method],
                            epoch,
                            best_val_text,
                            f"{_pct(record['accuracy']):.2f}",
                            f"{record['training_seconds']:.1f}",
                            f"{record['peak_gpu_memory_mb']:.0f}",
                            f"{record['trainable_parameters']:,}",
                        ]
                    )
                    + r" \\"
                )
            if dataset != DATASETS[-1]:
                lines.append(r"\midrule")
        if family != FAMILIES[-1]:
            lines.append(r"\midrule")
    return "\n".join(lines) + "\n"


def latex_tables(runs: dict) -> None:
    (TAB / "results_vit.tex").write_text(_result_rows(runs, "vit"), encoding="utf-8")
    (TAB / "results_dino.tex").write_text(_result_rows(runs, "dino_v3"), encoding="utf-8")
    (TAB / "per_run.tex").write_text(_per_run_rows(runs), encoding="utf-8")
    print("wrote LaTeX table snippets")


def main() -> None:
    runs = _runs()
    pipeline_figure()
    methods_figure()
    dataset_samples()
    architecture_images()
    loss_curves(runs)
    accuracy_bars(runs)
    efficiency(runs)
    per_class_f1()
    confusion(runs)
    gradcam()
    tsne(runs)
    latex_tables(runs)


if __name__ == "__main__":
    main()
