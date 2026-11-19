from __future__ import annotations

from pathlib import Path

from src.dino_v3.report import (
    _write_csv,
    _write_markdown,
    collect_rows,
    plot_pareto,
    plot_tradeoffs,
)

RESULTS_DIR = Path("results")
MODEL_DIRS = {
    "vit": RESULTS_DIR / "vit",
    "dinov3": RESULTS_DIR / "dino_v3",
}
ANALYSIS_DIR = RESULTS_DIR / "analysis"

def main() -> None:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for model_family, results_root in MODEL_DIRS.items():
        if not results_root.is_dir():
            print(f"[analysis] no results in {results_root}, skipping")
            continue
        for row in collect_rows(results_root):
            row["model_family"] = model_family
            row["model"] = row.get("model") or model_family
            grouped.setdefault((row["dataset"], model_family), []).append(row)

    if not grouped:
        raise SystemExit(
            "No metrics.json files found under results/vit or results/dino_v3"
        )

    for (dataset, model_family), rows in sorted(grouped.items()):
        output_dir = ANALYSIS_DIR / dataset / model_family
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(rows, output_dir / "summary.csv")
        _write_markdown(rows, output_dir / "summary.md")
        plot_tradeoffs(rows, output_dir / "tradeoffs.png")
        plot_pareto(rows, output_dir / "accuracy_pareto.png")
        print(f"[analysis] {dataset}/{model_family}: {len(rows)} runs -> {output_dir}")

    print(f"[analysis] grouped results written under {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
