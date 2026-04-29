#!/usr/bin/env python3
"""Plot mean test metrics from one or more metrics_test_set.csv files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-idownscale")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mean test metrics from metrics_test_set.csv files.")
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "CSV"),
        required=True,
        help="Pair of label and metrics_test_set.csv path. Repeat for multiple runs.",
    )
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Mean test metrics comparison", help="Figure title")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["loss", "rmse", "mae"]
    rows = []
    for label, csv_path in args.run:
        df = pd.read_csv(csv_path)
        row = {"label": label}
        for metric in metrics:
            row[metric] = float(df[metric].mean()) if metric in df.columns else float("nan")
        rows.append(row)

    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=False)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]

    for ax, metric in zip(axes, metrics):
        values = summary[metric]
        ax.bar(summary["label"], values, color=colors[: len(summary)], width=0.7)
        ax.set_title(metric.upper())
        ax.set_ylabel("Mean value")
        ax.grid(True, axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
