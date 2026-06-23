#!/usr/bin/env python3
"""
Plot observation-workflow distribution comparisons for raw, BC, and one or more ML models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(".")

from iriscc.runtime_paths import require_existing_file, require_match, resolve_runtime_sample_dir, resolve_sample_file
from iriscc.settings import GRAPHS_DIR, METRICS_DIR, get_metrics_test_name, get_prediction_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw/BC/ML distribution comparisons for observation workflows.")
    parser.add_argument("--exp", required=True, help="Experiment name.")
    parser.add_argument("--simu", default="gcm", help="Raw simulation variant, e.g. gcm or rcm.")
    parser.add_argument("--simu-test", default="gcm_bc", help="Bias-corrected simulation variant, e.g. gcm_bc.")
    parser.add_argument("--var", default="tas", help="Variable name.")
    parser.add_argument("--startdate", required=True, help="YYYYMMDD start date.")
    parser.add_argument("--enddate", required=True, help="YYYYMMDD end date.")
    parser.add_argument("--ml-models", default="", help="Comma-separated ML test names to compare.")
    parser.add_argument("--stride-days", type=int, default=7, help="Date stride in days for distribution sampling.")
    return parser.parse_args()


def smooth_density(
    values: np.ndarray,
    *,
    bins: int = 100,
    sigma_bins: float = 1.8,
) -> tuple[np.ndarray, np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot build a distribution plot from an empty array.")
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if np.isclose(lo, hi):
        hi = lo + 1.0
    hist, edges = np.histogram(finite, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma_bins**2))
    kernel /= kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    return centers, smooth


def collect_sample_distribution(dataset_dir: Path, key: str, index: int, dates: list[pd.Timestamp]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for date in dates:
        sample = resolve_sample_file(dataset_dir, date.strftime("%Y%m%d"))
        data = np.load(sample)
        arr = data[key][index].astype(np.float64)
        chunks.append(arr.ravel())
    return np.concatenate(chunks)


def collect_prediction_distribution(path: Path, dates: list[pd.Timestamp]) -> np.ndarray:
    ds = xr.open_dataset(path)
    data_var = next(iter(ds.data_vars))
    target_times = pd.to_datetime(ds.time.values)
    wanted = pd.DatetimeIndex(dates)
    indexer = target_times.get_indexer(wanted)
    if np.any(indexer < 0):
        missing = wanted[indexer < 0]
        raise FileNotFoundError(f"Missing {len(missing)} prediction dates in {path.name}: {missing[0]}")
    arr = ds[data_var].isel(time=indexer).values.astype(np.float64)
    ds.close()
    return arr.ravel()


def resolve_prediction_source(exp: str, simu_test: str, var: str, startdate: str, enddate: str, prediction_name: str) -> Path:
    exact = get_prediction_output_path(exp, simu_test, var, startdate, enddate, prediction_name)
    if exact.exists():
        return exact
    candidates = require_match(
        exact.parent,
        f"{var}_*_{startdate[:4]}*_{exp}_{prediction_name}.nc",
        f"prediction file for {prediction_name}",
        allow_multiple=True,
    )
    requested_start = pd.Timestamp(startdate)
    requested_end = pd.Timestamp(enddate)
    for candidate in candidates:
        with xr.open_dataset(candidate) as ds:
            times = pd.to_datetime(ds.time.values)
            if len(times) and times[0] <= requested_start and times[-1] >= requested_end:
                return candidate
    return require_existing_file(exact, f"prediction file for {prediction_name}")


def model_slug(models: list[str]) -> str:
    if not models:
        return "no_ml"
    return "__".join(model.replace("/", "_") for model in models)


def main() -> None:
    args = parse_args()
    ml_models = [item.strip() for item in args.ml_models.split(",") if item.strip()]

    dates = list(pd.date_range(args.startdate, args.enddate, freq=f"{args.stride_days}D"))
    raw_dir = resolve_runtime_sample_dir(args.exp, f"{args.simu}_raw")
    bc_dir = resolve_runtime_sample_dir(args.exp, "baseline", simu_test=args.simu_test)

    target = collect_sample_distribution(bc_dir, "y", 0, dates)
    raw = collect_sample_distribution(raw_dir, "x", -1, dates)
    bc = collect_sample_distribution(bc_dir, "x", -1, dates)

    series = [
        ("Target", target, "black"),
        (f"{args.simu.upper()} raw", raw, "dimgray"),
        (f"{args.simu_test.upper()} BC", bc, "tab:blue"),
    ]
    ml_colors = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
    summary_rows = []
    for idx, model in enumerate(ml_models):
        prediction_name = get_metrics_test_name(model, args.simu_test)
        prediction_path = resolve_prediction_source(
            args.exp,
            args.simu_test,
            args.var,
            args.startdate,
            args.enddate,
            prediction_name,
        )
        values = collect_prediction_distribution(prediction_path, dates)
        series.append((model, values, ml_colors[idx % len(ml_colors)]))

    for label, values, _ in series:
        summary_rows.append(
            {
                "curve": label,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "count": int(np.isfinite(values).sum()),
            }
        )

    fig, ax = plt.subplots(figsize=(11, 7))
    for label, values, color in series:
        centers, density = smooth_density(values)
        ax.plot(centers, density, color=color, label=label, linewidth=2.3)
        ax.axvline(np.nanmean(values), color=color, linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"{args.var} [K]")
    ax.set_ylabel("Density")
    ax.set_title(f"{args.exp} historical distribution comparison ({args.startdate}-{args.enddate}, stride {args.stride_days}d)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)

    out_dir = Path(GRAPHS_DIR) / "metrics" / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.exp}_{args.simu}_{args.simu_test}_{args.startdate}_{args.enddate}_{model_slug(ml_models)}"
    png = out_dir / f"distribution_compare_{stem}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    csv = Path(METRICS_DIR) / args.exp / f"distribution_compare_{stem}.csv"
    pd.DataFrame(summary_rows).to_csv(csv, index=False)
    print(f"png={png}")
    print(f"csv={csv}")


if __name__ == "__main__":
    main()
