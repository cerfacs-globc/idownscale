#!/usr/bin/env python3
"""Compare a training run test-metrics CSV against an archive reference CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare mean metrics from a training-run metrics_test_set.csv against an archive reference."
    )
    parser.add_argument("--current", required=True, help="Path to current metrics_test_set.csv")
    parser.add_argument("--archive", required=True, help="Path to archive/reference metrics_test_set.csv")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["loss", "rmse", "mae"],
        help="Metric columns to compare (default: loss rmse mae)",
    )
    return parser.parse_args()


def summarize(path: Path, metrics: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    df = pd.read_csv(path)
    summary: dict[str, float] = {"n_rows": float(len(df))}
    for metric in metrics:
        if metric in df.columns:
            summary[metric] = float(df[metric].mean())
    return df, summary


def main() -> int:
    args = parse_args()
    current_path = Path(args.current)
    archive_path = Path(args.archive)

    current_df, current_summary = summarize(current_path, args.metrics)
    archive_df, archive_summary = summarize(archive_path, args.metrics)

    print("Current:", current_path)
    print("Archive:", archive_path)
    print()
    print(f"{'metric':<12} {'current':>14} {'archive':>14} {'delta':>14}")
    print("-" * 58)
    for metric in ["n_rows", *args.metrics]:
        if metric not in current_summary and metric not in archive_summary:
            continue
        cur = current_summary.get(metric, float("nan"))
        arc = archive_summary.get(metric, float("nan"))
        delta = cur - arc
        print(f"{metric:<12} {cur:14.6f} {arc:14.6f} {delta:14.6f}")

    extra_current = sorted(set(current_df.columns) - set(args.metrics))
    extra_archive = sorted(set(archive_df.columns) - set(args.metrics))
    print()
    print("Current columns:", ", ".join(current_df.columns))
    print("Archive columns:", ", ".join(archive_df.columns))
    if extra_current != extra_archive:
        print("Column-set difference detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
