#!/usr/bin/env python3
"""
Build a historical 5-curve PDF comparison for exp5 (2000-2014).

This version deliberately reuses the workflow-ready sample datasets instead of
re-reading very large raw NetCDF sources. That keeps the figure tied to the
validated pipeline products and makes it practical to regenerate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(".")

from iriscc.settings import DATASET_BC_DIR, EXP5_ARCHIVE_DATASET_DIR, GRAPHS_DIR, PREDICTION_DIR


EXP = "exp5"
START = "2000-01-01"
END = "2014-12-31"
DAY_STEP = 7
INPUT_CHANNEL = 1


def iter_sample_dates() -> list[pd.Timestamp]:
    return list(pd.date_range(START, END, freq=f"{DAY_STEP}D"))


def collect_from_samples(dataset_dir: Path, key: str, index: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for date in iter_sample_dates():
        sample = dataset_dir / f"sample_{date.strftime('%Y%m%d')}.npz"
        if not sample.exists():
            continue
        data = np.load(sample)
        arr = data[key][index].astype(np.float64)
        chunks.append(arr.ravel())
    if not chunks:
        raise FileNotFoundError(f"No sample chunks collected from {dataset_dir}")
    return np.concatenate(chunks)


def collect_prediction() -> np.ndarray:
    path = (
        PREDICTION_DIR
        / "tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_all_gcm_bc.nc"
    )
    ds = xr.open_dataset(path)
    data_var = next(iter(ds.data_vars))
    arr = ds[data_var].isel(time=slice(None, None, DAY_STEP)).values.astype(np.float64)
    ds.close()
    return arr.ravel()


def smooth_density(
    values: np.ndarray,
    *,
    bins: int = 100,
    value_range: tuple[float, float] = (260.0, 310.0),
    sigma_bins: float = 1.8,
) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values[np.isfinite(values)], bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x**2) / (2 * sigma_bins**2))
    kernel /= kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    return centers, smooth


def main() -> None:
    raw_dir = DATASET_BC_DIR / "dataset_exp5_test_gcm"
    bc_dir = DATASET_BC_DIR / "dataset_exp5_test_gcm_bc"
    era5_dir = EXP5_ARCHIVE_DATASET_DIR

    raw = collect_from_samples(raw_dir, "x", INPUT_CHANNEL)
    bc = collect_from_samples(bc_dir, "x", INPUT_CHANNEL)
    era5 = collect_from_samples(era5_dir, "x", INPUT_CHANNEL)
    eobs = collect_from_samples(era5_dir, "y", 0)
    ai = collect_prediction()

    fig, ax = plt.subplots(figsize=(12, 7))
    series = [
        ("Raw GCM", raw, "gray"),
        ("GCM-BC", bc, "blue"),
        ("UNet", ai, "red"),
        ("ERA5", era5, "green"),
        ("E-OBS", eobs, "black"),
    ]
    for label, values, color in series:
        centers, density = smooth_density(values)
        ax.plot(centers, density, color=color, label=label, linewidth=2.5)
        ax.axvline(np.nanmean(values), color=color, linestyle="--", linewidth=2)
    ax.set_ylim(0, 0.07)
    ax.set_xlabel("Temperature [K]")
    ax.legend(loc="upper right")
    ax.set_title(
        f"exp5 Historical Distribution Comparison (2000-2014, {DAY_STEP}-day stride sample)",
        fontsize=16,
    )

    out_dir = Path(GRAPHS_DIR) / "metrics" / EXP
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp5_historical_5curve_pdf.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
