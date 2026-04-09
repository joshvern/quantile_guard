#!/usr/bin/env python
"""
Generate benchmark report tables and figures from raw CSV results.

Usage:
    python benchmarks/report.py
    python benchmarks/report.py --input benchmarks/results/raw/linear_baselines.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    """Load benchmark CSV."""
    return pd.read_csv(path)


def generate_accuracy_table(df: pd.DataFrame) -> str:
    """Generate markdown table: pinball loss by model and dataset size."""
    pivot = df.pivot_table(
        index=["n", "n_taus"],
        columns="model",
        values="mean_pinball_loss",
        aggfunc="first",
    )
    pivot = pivot.round(4)

    lines = ["## Pinball Loss Comparison", ""]
    lines.append("| n | quantiles | " + " | ".join(pivot.columns) + " |")
    lines.append("|---:|---:|" + "|".join(["---:" for _ in pivot.columns]) + "|")
    for (n, ntau), row in pivot.iterrows():
        vals = " | ".join(f"{v:.4f}" if not np.isnan(v) else "—" for v in row)
        lines.append(f"| {n} | {ntau} | {vals} |")
    lines.append("")
    return "\n".join(lines)


def generate_timing_table(df: pd.DataFrame) -> str:
    """Generate markdown table: fit time by model and dataset size."""
    pivot = df.pivot_table(
        index=["n", "n_taus"],
        columns="model",
        values="fit_time_s",
        aggfunc="first",
    )
    pivot = pivot.round(4)

    lines = ["## Fit Time (seconds)", ""]
    lines.append("| n | quantiles | " + " | ".join(pivot.columns) + " |")
    lines.append("|---:|---:|" + "|".join(["---:" for _ in pivot.columns]) + "|")
    for (n, ntau), row in pivot.iterrows():
        vals = " | ".join(f"{v:.4f}" if not np.isnan(v) else "—" for v in row)
        lines.append(f"| {n} | {ntau} | {vals} |")
    lines.append("")
    return "\n".join(lines)


def generate_crossing_table(df: pd.DataFrame) -> str:
    """Generate markdown table: crossing rate by model and dataset size."""
    pivot = df.pivot_table(
        index=["n", "n_taus"],
        columns="model",
        values="crossing_rate",
        aggfunc="first",
    )
    pivot = pivot.round(4)

    lines = ["## Crossing Rate", ""]
    lines.append("| n | quantiles | " + " | ".join(pivot.columns) + " |")
    lines.append("|---:|---:|" + "|".join(["---:" for _ in pivot.columns]) + "|")
    for (n, ntau), row in pivot.iterrows():
        vals = " | ".join(f"{v:.4f}" if not np.isnan(v) else "—" for v in row)
        lines.append(f"| {n} | {ntau} | {vals} |")
    lines.append("")
    return "\n".join(lines)


def generate_coverage_table(df: pd.DataFrame) -> str:
    """Generate markdown table: coverage by model and dataset size."""
    pivot = df.pivot_table(
        index=["n", "n_taus"],
        columns="model",
        values="empirical_coverage",
        aggfunc="first",
    )
    pivot = pivot.round(4)

    lines = ["## Empirical Coverage (outer quantile interval)", ""]
    lines.append("| n | quantiles | " + " | ".join(pivot.columns) + " |")
    lines.append("|---:|---:|" + "|".join(["---:" for _ in pivot.columns]) + "|")
    for (n, ntau), row in pivot.iterrows():
        vals = " | ".join(f"{v:.4f}" if not np.isnan(v) else "—" for v in row)
        lines.append(f"| {n} | {ntau} | {vals} |")
    lines.append("")
    return "\n".join(lines)


def generate_figures(df: pd.DataFrame, output_dir: Path):
    """Generate benchmark charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    quantile_counts = sorted(df["n_taus"].unique())

    def _iter_axes(figsize_scale=6):
        fig, axes = plt.subplots(
            1,
            len(quantile_counts),
            figsize=(figsize_scale * len(quantile_counts), 5),
            squeeze=False,
            sharey=len(quantile_counts) > 1,
        )
        return fig, axes.flatten()

    # 1. Fit time by dataset size
    fig, axes = _iter_axes()
    for ax, n_taus in zip(axes, quantile_counts):
        subset_nt = df[df["n_taus"] == n_taus]
        for model in subset_nt["model"].unique():
            subset = subset_nt[subset_nt["model"] == model].sort_values("n")
            ax.plot(subset["n"], subset["fit_time_s"], "o-", label=model)
        ax.set_xlabel("Dataset size (n)")
        ax.set_ylabel("Fit time (seconds)")
        ax.set_title(f"Fit Time vs Dataset Size ({n_taus} quantiles)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fit_time_vs_n.png", dpi=150)
    plt.close(fig)

    # 2. Pinball loss comparison
    fig, axes = _iter_axes()
    for ax, n_taus in zip(axes, quantile_counts):
        subset_nt = df[df["n_taus"] == n_taus]
        for model in subset_nt["model"].unique():
            subset = subset_nt[subset_nt["model"] == model].sort_values("n")
            ax.plot(subset["n"], subset["mean_pinball_loss"], "o-", label=model)
        ax.set_xlabel("Dataset size (n)")
        ax.set_ylabel("Mean pinball loss")
        ax.set_title(f"Pinball Loss vs Dataset Size ({n_taus} quantiles)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pinball_loss_vs_n.png", dpi=150)
    plt.close(fig)

    # 3. Crossing rate bar chart
    fig, axes = _iter_axes(figsize_scale=7)
    for ax, n_taus in zip(axes, quantile_counts):
        cross_df = df[df["n_taus"] == n_taus].copy()
        if len(cross_df) == 0:
            continue

        models = list(cross_df["model"].unique())
        sizes = sorted(cross_df["n"].unique())
        x = np.arange(len(sizes))
        width = 0.8 / max(len(models), 1)

        for i, model in enumerate(models):
            subset = cross_df[cross_df["model"] == model]
            heights = [
                subset.loc[subset["n"] == size, "crossing_rate"].iloc[0]
                if np.any(subset["n"] == size)
                else np.nan
                for size in sizes
            ]
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar(x + offset, heights, width, label=model)

        ax.set_xlabel("Dataset size (n)")
        ax.set_ylabel("Crossing rate")
        ax.set_title(f"Crossing Rate by Model ({n_taus} quantiles)")
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "crossing_rate.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument(
        "--input",
        default="benchmarks/results/raw/linear_baselines.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--tables-dir",
        default="benchmarks/results/tables",
        help="Output directory for markdown tables",
    )
    parser.add_argument(
        "--figures-dir",
        default="benchmarks/results/figures",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    df = load_results(args.input)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Generate tables
    accuracy = generate_accuracy_table(df)
    timing = generate_timing_table(df)
    crossing = generate_crossing_table(df)
    coverage = generate_coverage_table(df)

    full_report = "\n".join([
        "# Benchmark Results: Linear Baselines",
        "",
        f"Generated from `{args.input}`",
        "",
        accuracy,
        timing,
        crossing,
        coverage,
    ])

    report_path = tables_dir / "linear_baselines.md"
    report_path.write_text(full_report)
    print(f"Report written to {report_path}")

    # Generate figures
    generate_figures(df, figures_dir)

    return full_report


if __name__ == "__main__":
    main()
