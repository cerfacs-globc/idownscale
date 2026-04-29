#!/usr/bin/env python3
"""Compare two historical exp5 evaluation outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare historical evaluation products for two exp5 runs.")
    parser.add_argument("--baseline", required=True, help="Baseline test-name stem, e.g. unet_all_gcm_bc")
    parser.add_argument("--candidate", required=True, help="Candidate test-name stem, e.g. unet_grace30_gcm_bc")
    parser.add_argument(
        "--metrics-root",
        default=os.getenv("IDOWNSCALE_METRICS_EXP5_ROOT", "metrics/exp5"),
        help="Root exp5 metrics directory",
    )
    return parser.parse_args()


def compare_mean_csv(path_a: Path, path_b: Path, label: str) -> list[str]:
    if not path_a.exists() or not path_b.exists():
        return [f"\n[{label}]", f"  skipped: missing file(s) baseline_exists={path_a.exists()} candidate_exists={path_b.exists()}"]
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    key_col = df_a.columns[0]
    merged = df_a.merge(df_b, on=key_col, suffixes=("_baseline", "_candidate"))
    lines = [f"\n[{label}]"]
    for _, row in merged.iterrows():
        group = row[key_col]
        lines.append(f"  {group}:")
        for col in df_a.columns[1:]:
            base = float(row[f"{col}_baseline"])
            cand = float(row[f"{col}_candidate"])
            lines.append(f"    {col}: baseline={base:.6f} candidate={cand:.6f} delta={cand-base:+.6f}")
    return lines


def compare_value_csv(path_a: Path, path_b: Path) -> list[str]:
    if not path_a.exists() or not path_b.exists():
        return [f"\n[value_metrics]", f"  skipped: missing file(s) baseline_exists={path_a.exists()} candidate_exists={path_b.exists()}"]
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    row_a = df_a.iloc[0]
    row_b = df_b.iloc[0]
    lines = ["\n[value_metrics]"]
    for col in df_a.columns:
        a = row_a[col]
        b = row_b[col]
        if pd.isna(a) and pd.isna(b):
            delta = float("nan")
        else:
            delta = float(b) - float(a)
        a_str = "nan" if pd.isna(a) else f"{float(a):.6f}"
        b_str = "nan" if pd.isna(b) else f"{float(b):.6f}"
        d_str = "nan" if np.isnan(delta) else f"{delta:+.6f}"
        lines.append(f"  {col}: baseline={a_str} candidate={b_str} delta={d_str}")
    return lines


def compare_npz(path_a: Path, path_b: Path, label: str) -> list[str]:
    if not path_a.exists() or not path_b.exists():
        return [f"\n[{label}]", f"  skipped: missing file(s) baseline_exists={path_a.exists()} candidate_exists={path_b.exists()}"]
    npz_a = dict(np.load(path_a, allow_pickle=True))
    npz_b = dict(np.load(path_b, allow_pickle=True))
    lines = [f"\n[{label}]"]
    for key in sorted(npz_a.keys()):
        if key not in npz_b:
            lines.append(f"  {key}: missing in candidate")
            continue
        a = np.asarray(npz_a[key])
        b = np.asarray(npz_b[key])
        if a.shape != b.shape:
            lines.append(f"  {key}: shape mismatch baseline={a.shape} candidate={b.shape}")
            continue
        if np.issubdtype(a.dtype, np.datetime64) or a.dtype.kind in {"U", "S", "O"}:
            same = np.array_equal(a, b)
            lines.append(f"  {key}: exact_match={same}")
            continue
        diff = np.nanmax(np.abs(a - b))
        lines.append(f"  {key}: max_abs_diff={float(diff):.6f}")
    return lines


def main() -> int:
    args = parse_args()
    root = Path(args.metrics_root)
    mean_root = root / "mean_metrics"

    outputs: list[str] = []
    outputs += compare_mean_csv(
        mean_root / f"metrics_test_mean_daily_exp5_{args.baseline}.csv",
        mean_root / f"metrics_test_mean_daily_exp5_{args.candidate}.csv",
        "mean_daily_csv",
    )
    outputs += compare_mean_csv(
        mean_root / f"metrics_test_mean_monthly_exp5_{args.baseline}.csv",
        mean_root / f"metrics_test_mean_monthly_exp5_{args.candidate}.csv",
        "mean_monthly_csv",
    )
    outputs += compare_value_csv(
        root / "value_metrics_exp5_unet_all.csv",
        root / "value_metrics_exp5_unet_grace30.csv",
    )
    outputs += compare_npz(
        mean_root / f"metrics_test_daily_exp5_{args.baseline}.npz",
        mean_root / f"metrics_test_daily_exp5_{args.candidate}.npz",
        "daily_npz",
    )
    outputs += compare_npz(
        mean_root / f"metrics_test_monthly_exp5_{args.baseline}.npz",
        mean_root / f"metrics_test_monthly_exp5_{args.candidate}.npz",
        "monthly_npz",
    )
    print("\n".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
