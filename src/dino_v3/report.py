from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .config import METHODS

COLUMNS = [
    "dataset",
    "model_family",
    "model",
    "method",
    "seed",
    "shots_per_class",
    "accuracy",
    "top5_accuracy",
    "macro_f1",
    "weighted_f1",
    "trainable_parameters",
    "trainable_fraction",
    "training_seconds",
    "peak_gpu_memory_mb",
    "epochs",
]


def collect_rows(results_dir: str | Path) -> list[dict]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(results_dir).glob("*/*/seed_*/metrics.json"))
    ]
    order = {method: index for index, method in enumerate(METHODS)}
    return sorted(
        rows,
        key=lambda row: (row["dataset"], order.get(row["method"], 99), row["seed"]),
    )


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in COLUMNS} for row in rows)


def _write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "| dataset | model | method | shots | top-1 acc | top-5 acc | macro F1 | "
        "weighted F1 | trainable params | % trainable | train time (s) | peak GPU (MB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {model} | {method} | {shots} | {acc:.4f} | {top5:.4f} | "
            "{macro:.4f} | {weighted:.4f} | {params:,} | {fraction:.2f} | {seconds:.1f} | "
            "{memory} |".format(
                dataset=row["dataset"],
                model=row.get("model") or "dinov3",
                method=row["method"],
                shots=row.get("shots_per_class") or "full",
                acc=row["accuracy"],
                top5=row["top5_accuracy"],
                macro=row["macro_f1"],
                weighted=row["weighted_f1"],
                params=row["trainable_parameters"],
                fraction=100 * row["trainable_fraction"],
                seconds=row["training_seconds"],
                memory=(
                    f"{row['peak_gpu_memory_mb']:.0f}"
                    if row.get("peak_gpu_memory_mb") is not None
                    else "n/a"
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_labels(rows: list[dict]) -> list[str]:
    datasets = {row["dataset"] for row in rows}
    labels = []
    for row in rows:
        prefix = f"{row['dataset']}\n" if len(datasets) > 1 else ""
        labels.append(f"{prefix}{row['method']}\n{row['shots_per_class'] or 'full'} sh")
    return labels


def _annotate_bars(axis, bars, values, formatter) -> None:
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_tradeoffs(rows: list[dict], path: Path) -> None:
    labels = _method_labels(rows)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))

    accuracies = [row["accuracy"] for row in rows]
    f1_scores = [row["macro_f1"] for row in rows]
    parameters = [row["trainable_parameters"] for row in rows]

    accuracy_bars = axes[0].bar(labels, accuracies, color="#4C72B0")
    axes[0].set_title("Held-out top-1 accuracy")
    axes[0].set_ylim(0, 1.12)
    _annotate_bars(axes[0], accuracy_bars, accuracies, lambda value: f"{value:.4f}")

    f1_bars = axes[1].bar(labels, f1_scores, color="#55A868")
    axes[1].set_title("Held-out macro F1")
    axes[1].set_ylim(0, 1.12)
    _annotate_bars(axes[1], f1_bars, f1_scores, lambda value: f"{value:.4f}")

    parameter_bars = axes[2].bar(
        labels, [max(value, 1) for value in parameters], color="#C44E52"
    )
    axes[2].set_title("Trainable parameters")
    axes[2].set_yscale("log")
    _annotate_bars(
        axes[2],
        parameter_bars,
        parameters,
        lambda value: f"{value:,}" if value else "0",
    )

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=30)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_pareto(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P"]

    lowest = min(row["accuracy"] for row in rows)
    y_floor = max(0.0, lowest - 0.02)
    y_ceiling = min(1.005, max(row["accuracy"] for row in rows) + 0.005)

    for axis, x_key, x_label, log_x in (
        (axes[0], "trainable_parameters", "trainable parameters", True),
        (axes[1], "training_seconds", "training time (s)", False),
    ):
        for index, row in enumerate(rows):
            axis.scatter(
                max(row[x_key], 1e-3),
                row["accuracy"],
                s=80,
                color=colors[index % len(colors)],
                marker=markers[index % len(markers)],
                label=row["method"],
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel("held-out top-1 accuracy")
        axis.set_ylim(y_floor, y_ceiling)
        axis.grid(alpha=0.3)
        if log_x:
            axis.set_xscale("log")

    axes[0].set_title("Accuracy vs. trainable parameters")
    axes[1].set_title("Accuracy vs. training time")
    axes[1].legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_report(results_dir: str | Path) -> list[dict]:
    results_dir = Path(results_dir)
    rows = collect_rows(results_dir)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {results_dir}")

    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, analysis_dir / "summary.csv")
    _write_markdown(rows, analysis_dir / "summary.md")
    plot_tradeoffs(rows, analysis_dir / "tradeoffs.png")
    plot_pareto(rows, analysis_dir / "accuracy_pareto.png")
    print(f"[report] {len(rows)} runs aggregated into {analysis_dir}")
    return rows
