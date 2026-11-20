#!/usr/bin/env python3
"""Build the figures used by the TPA-17 beamer deck.

Everything is derived from the artefacts in ``results/`` and the raw images in
``data/``; no model training is required.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
DECK = Path(__file__).resolve().parent
FIG = DECK / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = ["flowers102", "oxfordiiitpet"]
FAMILIES = ["vit", "dino_v3"]
METHODS = ["full", "lp", "lora", "bitfit", "ncm"]
TRAINED = ["full", "lp", "lora", "bitfit"]

PRETTY_DS = {"flowers102": "Flowers102", "oxfordiiitpet": "Oxford-IIIT Pet"}
PRETTY_FAM = {"vit": "ViT-B/16", "dino_v3": "DINOv3-B/16"}
PRETTY_METHOD = {
    "full": "Full",
    "lp": "Linear probe",
    "lora": "LoRA",
    "bitfit": "BitFit",
    "ncm": "NCM (training-free)",
}
SHORT_METHOD = {"full": "Full", "lp": "LP", "lora": "LoRA", "bitfit": "BitFit", "ncm": "NCM"}
COLORS = {"full": "#1f77b4", "lp": "#2ca02c", "lora": "#d62728", "bitfit": "#9467bd", "ncm": "#ff7f0e"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 160,
    }
)


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
def _family_key(record: dict, path: Path) -> str:
    raw = record.get("model_family") or path.parents[3].name
    return "dino_v3" if raw == "dinov3" else raw


def load_runs() -> dict[tuple[str, str, str], dict]:
    runs: dict[tuple[str, str, str], dict] = {}
    for path in sorted((ROOT / "results").glob("*/*/*/seed_*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_family"] = _family_key(record, path)
        runs[(record["_family"], record["dataset"], record["method"])] = record
    return runs


def aggregate_precision_recall(record: dict) -> tuple[float, float]:
    family = record["_family"]
    path = (
        ROOT
        / "results"
        / family
        / record["dataset"]
        / record["method"]
        / f"seed_{record['seed']}"
        / "per_class_f1.csv"
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    precision = np.mean([float(r["precision"]) for r in rows])
    recall = np.mean([float(r["recall"]) for r in rows])
    return float(precision), float(recall)


def write_summary(runs: dict) -> None:
    path = FIG / "metrics_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "dataset",
                "method",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "macro_f1",
                "top5_accuracy",
                "trainable_parameters",
                "trainable_fraction",
                "training_seconds",
                "peak_gpu_memory_mb",
            ]
        )
        for family in FAMILIES:
            for dataset in DATASETS:
                for method in METHODS:
                    record = runs[(family, dataset, method)]
                    precision, recall = aggregate_precision_recall(record)
                    writer.writerow(
                        [
                            family,
                            dataset,
                            method,
                            f"{record['accuracy']:.6f}",
                            f"{precision:.6f}",
                            f"{recall:.6f}",
                            f"{record['macro_f1']:.6f}",
                            f"{record['top5_accuracy']:.6f}",
                            record["trainable_parameters"],
                            f"{record['trainable_fraction']:.6f}",
                            f"{record['training_seconds']:.1f}",
                            f"{record['peak_gpu_memory_mb']:.1f}",
                        ]
                    )


# --------------------------------------------------------------------------- #
# Loss / accuracy curves
# --------------------------------------------------------------------------- #
def parse_epochs(csv_path: Path) -> dict[int, dict[str, float]]:
    epochs: dict[int, dict[str, float]] = {}
    for row in csv.DictReader(csv_path.open(encoding="utf-8")):
        if not row["epoch"].strip():
            continue
        epoch = int(float(row["epoch"]))
        entry = epochs.setdefault(epoch, {})
        if row["train/loss_epoch"].strip():
            entry["train_loss"] = float(row["train/loss_epoch"])
        if row["val/loss"].strip():
            entry["val_loss"] = float(row["val/loss"])
        if row["val/acc"].strip():
            entry["val_acc"] = float(row["val/acc"])
    return {epoch: values for epoch, values in epochs.items() if "train_loss" in values or "val_loss" in values}


def plot_loss_curves(runs: dict, family: str, dataset: str) -> None:
    with plt.rc_context(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10, 3.7), sharex=True)
        for ax, method in zip(axes.ravel(), TRAINED):
            record = runs[(family, dataset, method)]
            checkpoint = Path(record["checkpoint"])
            epochs = parse_epochs(checkpoint.parents[1] / "metrics.csv")
            keys = sorted(epochs)
            train = [epochs[e].get("train_loss", np.nan) for e in keys]
            val = [epochs[e].get("val_loss", np.nan) for e in keys]
            acc = [epochs[e].get("val_acc", np.nan) for e in keys]

            ax2 = ax.twinx()
            line1, = ax.plot(keys, train, "o-", color="#1f77b4", markersize=3.2, label="train loss")
            line2, = ax.plot(keys, val, "s-", color="#d62728", markersize=3.2, label="val loss")
            line3, = ax2.plot(keys, acc, "--", color="#2ca02c", linewidth=1.6, label="val accuracy")
            ax.set_yscale("log")
            ax.set_title(f"{PRETTY_METHOD[method]}  (test acc {100 * record['accuracy']:.2f}%)")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss (log)")
            ax2.set_ylabel("val acc")
            if not np.all(np.isnan(acc)):
                low = np.nanmin(acc)
                ax2.set_ylim(max(0.0, low - 0.04), 1.0)
            ax.grid(alpha=0.25, which="both")
            ax.legend([line1, line2, line3], ["train", "val", "val acc"], loc="best")

        fig.tight_layout(pad=0.4)
        out = FIG / f"loss_{family}_{dataset}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


# --------------------------------------------------------------------------- #
# Dataset samples
# --------------------------------------------------------------------------- #
def _compose_samples(samples: list[tuple[Image.Image, str]], out: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.6), gridspec_kw={"hspace": 0.34, "wspace": 0.05})
    for ax, (image, label) in zip(axes.ravel(), samples):
        ax.imshow(image)
        ax.set_title(label, fontsize=10, pad=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.34)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def flowers_samples() -> None:
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
    _compose_samples(samples, FIG / "flowers_samples.png", "Flowers102: six of the 102 classes")


def pet_samples() -> None:
    chosen = ["beagle", "persian", "pug", "siamese", "boxer", "maine_coon"]
    samples = []
    for name in chosen:
        files = sorted((ROOT / "data/oxford_pet/test" / name).glob("*.jpg"))
        image = Image.open(files[len(files) // 2]).convert("RGB")
        image = Image.fromarray(np.asarray(image.resize((420, 320))))
        samples.append((image, name.replace("_", " ")))
    _compose_samples(samples, FIG / "pet_samples.png", "Oxford-IIIT Pet: six of the 37 breeds")


# --------------------------------------------------------------------------- #
# t-SNE maps
# --------------------------------------------------------------------------- #
def tsne_maps(runs: dict, family: str) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(16, 5.4))
    rng = np.random.default_rng(42)
    for row, dataset in enumerate(DATASETS):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            path = (
                ROOT
                / "results"
                / family
                / dataset
                / method
                / "seed_42"
                / "predictions.npz"
            )
            bundle = np.load(path)
            embeddings = bundle["val_embeddings"]
            labels = bundle["val_labels"]
            indices = rng.choice(
                len(labels), size=min(650, len(labels)), replace=False
            )
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
            ax.scatter(
                projected[:, 0],
                projected[:, 1],
                c=labels[indices],
                cmap="turbo",
                s=3,
                vmin=0,
                vmax=num_classes - 1,
                linewidths=0,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(PRETTY_METHOD[method], fontsize=10)
        axes[row, 0].set_ylabel(
            f"{PRETTY_DS[dataset]}\n({102 if dataset == 'flowers102' else 37} classes)",
            fontsize=10,
        )
    fig.suptitle(
        f"t-SNE of validation CLS embeddings: {PRETTY_FAM[family]}", fontsize=13
    )
    fig.tight_layout()
    out = FIG / f"tsne_{family}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Grad-CAM strips
# --------------------------------------------------------------------------- #
def _gradcam_panels(path: Path, panels: int = 4) -> list[Image.Image]:
    array = np.asarray(Image.open(path).convert("RGB"))
    height, width, _ = array.shape
    top = array[: int(height * 0.54)]
    mask = top.min(axis=2) < 245
    column_activity = mask.sum(axis=0)
    runs: list[tuple[int, int]] = []
    start = None
    for x, value in enumerate(column_activity > 3):
        if value and start is None:
            start = x
        elif not value and start is not None:
            if x - start > 40:
                runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(column_activity)))
    if len(runs) < panels:
        raise RuntimeError(f"Could not find {panels} panels in {path}: {runs}")
    runs = runs[:panels]
    y0 = int(height * 0.035)
    y1 = int(height * 0.515)
    return [Image.fromarray(array[y0:y1, x0:x1]) for x0, x1 in runs]


def gradcam_examples(runs: dict) -> None:
    columns = [
        ("flowers102", "full"),
        ("flowers102", "lp"),
        ("oxfordiiitpet", "full"),
        ("oxfordiiitpet", "lp"),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(12, 4.0))
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
                path = FIG / f"gradcam_dino_{method}_{dataset}.png"
            panel = _gradcam_panels(path)[0]
            ax = axes[row, col]
            ax.imshow(panel)
            ax.axis("off")
            if row == 0:
                ax.set_title(
                    f"{PRETTY_DS[dataset]} / {PRETTY_METHOD[method]}", fontsize=10
                )
        figure.text(
            0.012,
            0.72 - 0.42 * row,
            PRETTY_FAM[family],
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
        )
    figure.tight_layout(pad=0.25, rect=(0.03, 0, 1, 1))
    out = FIG / "gradcam_examples.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Confusion matrices
# --------------------------------------------------------------------------- #
def confusion_grid(runs: dict) -> None:
    specs = [
        ("vit", "flowers102", "ncm"),
        ("dino_v3", "flowers102", "ncm"),
        ("vit", "oxfordiiitpet", "full"),
        ("dino_v3", "oxfordiiitpet", "ncm"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 4.6))
    figure.subplots_adjust(left=0.07, right=0.87, bottom=0.13, top=0.90, wspace=0.22, hspace=0.55)
    image = None
    for ax, (family, dataset, method) in zip(axes.ravel(), specs):
        matrix = np.loadtxt(
            ROOT
            / "results"
            / family
            / dataset
            / method
            / "seed_42"
            / "confusion_matrix.csv",
            delimiter=",",
        )
        normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        accuracy = runs[(family, dataset, method)]["accuracy"] * 100
        short_dataset = "Flowers102" if dataset == "flowers102" else "Pet"
        ax.set_title(
            f"{PRETTY_FAM[family]} / {short_dataset} / {SHORT_METHOD[method]} "
            f"({accuracy:.2f}%)",
            fontsize=9,
        )
        ax.set_xlabel("predicted class", fontsize=8)
        ax.set_ylabel("true class", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar_axis = figure.add_axes([0.90, 0.17, 0.015, 0.66])
    figure.colorbar(image, cax=colorbar_axis, label="row-normalised rate")
    out = FIG / "confusion_grid.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Accuracy / parameter efficiency
# --------------------------------------------------------------------------- #
def efficiency_plot(runs: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    markers = {"vit": "o", "dino_v3": "s"}
    colors = {"vit": "#4c72b0", "dino_v3": "#c44e52"}
    for ax, dataset in zip(axes, DATASETS):
        for family in FAMILIES:
            for method in METHODS:
                record = runs[(family, dataset, method)]
                x = max(record["trainable_parameters"], 1)
                y = 100 * record["accuracy"]
                ax.scatter(
                    x,
                    y,
                    s=80,
                    marker=markers[family],
                    color=colors[family],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                )
                ax.annotate(
                    SHORT_METHOD[method],
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7.5,
                )
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters (log scale)")
        ax.set_ylabel("test top-1 accuracy (%)")
        ax.set_title(PRETTY_DS[dataset])
        ax.grid(alpha=0.25)
        low = min(100 * runs[(f, dataset, m)]["accuracy"] for f in FAMILIES for m in METHODS)
        high = max(100 * runs[(f, dataset, m)]["accuracy"] for f in FAMILIES for m in METHODS)
        ax.set_ylim(low - 0.4, high + 0.4)
        ax.scatter([], [], marker="o", color=colors["vit"], label="ViT-B/16")
        ax.scatter([], [], marker="s", color=colors["dino_v3"], label="DINOv3-B/16")
        ax.legend(loc="lower right", frameon=True)
        ax.text(
            0.02,
            0.03,
            "NCM has 0 trainable parameters",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#666666",
        )
    figure.tight_layout()
    out = FIG / "efficiency.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Architecture images
# --------------------------------------------------------------------------- #
def architecture_images() -> None:
    vit_source = ROOT / "archs" / "vit_arch.png"
    vit_target = FIG / "vit_arch.png"
    Image.open(vit_source).convert("RGB").save(vit_target)

    dino_source = ROOT / "archs" / "dino_v3.webp"
    dino_target = FIG / "dino_v3.png"
    image = Image.open(dino_source).convert("RGBA")
    canvas = Image.new("RGB", image.size, "white")
    canvas.paste(image, mask=image.split()[-1])
    canvas.save(dino_target)
    print("wrote", vit_target, "and", dino_target)


def main() -> None:
    runs = load_runs()
    write_summary(runs)
    architecture_images()
    flowers_samples()
    pet_samples()
    for family in FAMILIES:
        for dataset in DATASETS:
            plot_loss_curves(runs, family, dataset)
    for family in FAMILIES:
        tsne_maps(runs, family)
    gradcam_examples(runs)
    confusion_grid(runs)
    efficiency_plot(runs)


if __name__ == "__main__":
    main()
