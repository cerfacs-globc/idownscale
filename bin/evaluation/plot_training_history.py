#!/usr/bin/env python3
"""Plot training history from a Lightning CSV metrics file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-idownscale")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training history from Lightning CSV logs.")
    parser.add_argument("--metrics-csv", required=True, help="Path to Lightning metrics.csv")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Training history", help="Figure title")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path = Path(args.metrics_csv)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path)
    if df.empty:
        raise SystemExit(f"No rows found in {metrics_path}")

    train_df = df[df["train_loss"].notna()].copy() if "train_loss" in df else pd.DataFrame()
    val_df = df[df["val_loss"].notna()].copy() if "val_loss" in df else pd.DataFrame()
    epoch_df = df[df["epoch_time"].notna()].copy() if "epoch_time" in df else pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    if not train_df.empty:
        ax.plot(train_df["step"], train_df["train_loss"], marker="o", markersize=4, linewidth=1.8,
                color="#1f77b4", label="Train loss")
    if not val_df.empty:
        ax.plot(val_df["step"], val_df["val_loss"], marker="s", markersize=4, linewidth=1.8,
                color="#d62728", label="Val loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs step")
    ax.grid(True, alpha=0.25)
    if not train_df.empty or not val_df.empty:
        ax.legend(frameon=False)

    ax = axes[1]
    if not train_df.empty:
        ax.plot(train_df["epoch"], train_df["train_loss"], marker="o", markersize=5, linewidth=2.0,
                color="#1f77b4", label="Train loss")
    if not val_df.empty:
        ax.plot(val_df["epoch"], val_df["val_loss"], marker="s", markersize=5, linewidth=2.0,
                color="#d62728", label="Val loss")
        best_idx = val_df["val_loss"].idxmin()
        best_row = val_df.loc[best_idx]
        ax.scatter([best_row["epoch"]], [best_row["val_loss"]], color="#8c1d18", s=40, zorder=5)
        ax.annotate(
            f"best={best_row['val_loss']:.3f}\nepoch={int(best_row['epoch'])}",
            xy=(best_row["epoch"], best_row["val_loss"]),
            xytext=(8, -10),
            textcoords="offset points",
            fontsize=8,
            color="#8c1d18",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d9d9d9", alpha=0.9),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs epoch")
    ax.grid(True, alpha=0.25)
    if not train_df.empty or not val_df.empty:
        ax.legend(frameon=False)

    ax = axes[2]
    if not epoch_df.empty:
        ax.bar(epoch_df["epoch"].astype(int), epoch_df["epoch_time"], color="#2ca02c", width=0.75)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Seconds")
        ax.set_title("Epoch time")
        ax.grid(True, axis="y", alpha=0.25)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No epoch_time data", ha="center", va="center")

    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
