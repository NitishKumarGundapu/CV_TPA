from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--results-dir", default="outputs")
    parser.add_argument("--max-tsne-samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _plot_comparison(rows: list[dict], output: Path) -> None:
    labeled = [row for row in rows if row["accuracy"] is not None]
    labels = [f"{row['dataset']}\n{row['method']}" for row in labeled]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(labels, [row["accuracy"] for row in labeled])
    axes[0].set_title("Held-out top-1 accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].bar(labels, [row["trainable_parameters"] for row in labeled])
    axes[1].set_title("Trainable parameters")
    axes[1].set_yscale("log")
    axes[2].bar(labels, [row["training_seconds"] for row in labeled])
    axes[2].set_title("Training time (seconds)")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_tsne(
    archive: Path, output: Path, max_samples: int, seed: int
) -> None:
    arrays = np.load(archive)
    embeddings, targets = arrays["embeddings"], arrays["targets"]
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(targets), size=min(max_samples, len(targets)), replace=False
    )
    points = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(embeddings[indices])
    fig, axis = plt.subplots(figsize=(8, 7))
    scatter = axis.scatter(
        points[:, 0], points[:, 1], c=targets[indices], s=8, cmap="tab20"
    )
    axis.set_title(f"{archive.parent.parent.parent.name} / {archive.parent.parent.name}")
    axis.set_xticks([])
    axis.set_yticks([])
    fig.colorbar(scatter, ax=axis, label="class ID")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metric_files = sorted(results_dir.glob("*/*/seed_*/metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No experiment metrics found under {results_dir}")

    rows = [
        json.loads(path.read_text(encoding="utf-8")) for path in metric_files
    ]
    columns = list(rows[0])
    with (analysis_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    _plot_comparison(rows, analysis_dir / "tradeoffs.png")

    for metric_file in metric_files:
        archive = metric_file.with_name("predictions.npz")
        if archive.exists():
            row = json.loads(metric_file.read_text(encoding="utf-8"))
            _plot_tsne(
                archive,
                analysis_dir / f"tsne_{row['dataset']}_{row['method']}.png",
                args.max_tsne_samples,
                args.seed,
            )


if __name__ == "__main__":
    main()
