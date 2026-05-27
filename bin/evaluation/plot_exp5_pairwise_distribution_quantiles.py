#!/usr/bin/env python3
"""
Build a 4-panel historical comparison figure for exp5 (2000-2014).

Panels:
1. Raw GCM vs E-OBS
2. GCM-BC vs E-OBS
3. UNet vs E-OBS
4. Quantile bias summary (q5, q50, q95)
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


START = "2000-01-01"
END = "2014-12-31"
DAY_STEP = 7
INPUT_CHANNEL = 1
QUANTILES = [0.05, 0.50, 0.95]


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


def quantile_bias(values: np.ndarray, reference: np.ndarray) -> list[float]:
    return [float(np.nanquantile(values, q) - np.nanquantile(reference, q)) for q in QUANTILES]


def plot_difference(
    ax: plt.Axes,
    ref: np.ndarray,
    cmp_values: np.ndarray,
    cmp_label: str,
    cmp_color: str,
    *,
    median_bias: float,
) -> None:
    _, y_ref = smooth_density(ref)
    x_cmp, y_cmp = smooth_density(cmp_values)
    diff = y_cmp - y_ref
    ax.axhline(0.0, color="black", linewidth=1.5, alpha=0.8)
    ax.plot(x_cmp, diff, color=cmp_color, linewidth=2.5, label=f"{cmp_label} - E-OBS")
    ax.fill_between(x_cmp, 0.0, diff, color=cmp_color, alpha=0.15)
    ax.axvline(np.nanmean(ref), color="black", linestyle="--", linewidth=1.7, alpha=0.8)
    ax.axvline(np.nanmean(cmp_values), color=cmp_color, linestyle="--", linewidth=1.7)
    ax.set_xlim(260, 310)
    ax.set_ylim(-0.02, 0.02)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Density difference")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    ax.text(
        0.03,
        0.08,
        f"Median bias: {median_bias:+.2f} K",
        transform=ax.transAxes,
        fontsize=11.5,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3.0},
    )


def plot_quantile_panel(ax: plt.Axes, stats: dict[str, list[float]]) -> None:
    ax.set_axis_off()
    ax.set_title("PDF comparison summary", pad=6, fontsize=16)
    bar_ax = ax.inset_axes([0.05, 0.46, 0.82, 0.46])
    text_ax = ax.inset_axes([0.05, 0.02, 0.92, 0.34])
    text_ax.set_axis_off()

    labels = list(stats.keys())
    vals = np.array(list(stats.values()))
    colors = ["gray", "blue", "red"]
    x = np.arange(len(QUANTILES))
    width = 0.22
    offsets = [-width, 0.0, width]
    for idx, (label, color) in enumerate(zip(labels, colors)):
        bar_ax.bar(x + offsets[idx], vals[idx], width=width, color=color, alpha=0.85, label=label)
    bar_ax.axhline(0.0, color="black", linewidth=1.2)
    bar_ax.set_xticks(x, [f"q{int(q * 100)}" for q in QUANTILES])
    bar_ax.set_ylabel("Bias [K]")
    bar_ax.set_ylim(-1.6, 0.4)
    bar_ax.grid(axis="y", linestyle="--", alpha=0.3)
    bar_ax.legend(loc="lower left", ncol=3, frameon=False, fontsize=9)

    text = "\n\n".join(
        [
            "• BC improves the median shift vs Raw GCM.",
            "• Cold-tail bias remains the hardest case, especially for UNet.",
            "• Distribution agreement alone does not measure spatial realism.",
        ]
    )
    text_ax.text(0.0, 0.98, text, va="top", ha="left", fontsize=14.5, linespacing=1.45)


def main() -> None:
    raw_dir = DATASET_BC_DIR / "dataset_exp5_test_gcm"
    bc_dir = DATASET_BC_DIR / "dataset_exp5_test_gcm_bc"
    era5_dir = EXP5_ARCHIVE_DATASET_DIR

    raw = collect_from_samples(raw_dir, "x", INPUT_CHANNEL)
    bc = collect_from_samples(bc_dir, "x", INPUT_CHANNEL)
    eobs = collect_from_samples(era5_dir, "y", 0)
    ai = collect_prediction()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("exp5 Historical Distribution Validation (2000-2014, 7-day stride sample)", fontsize=20, y=0.98)

    raw_q = quantile_bias(raw, eobs)
    bc_q = quantile_bias(bc, eobs)
    ai_q = quantile_bias(ai, eobs)

    plot_difference(axes[0, 0], eobs, raw, "Raw GCM", "gray", median_bias=raw_q[1])
    axes[0, 0].set_title("Raw GCM - E-OBS")

    plot_difference(axes[0, 1], eobs, bc, "GCM-BC", "blue", median_bias=bc_q[1])
    axes[0, 1].set_title("GCM-BC - E-OBS")

    plot_difference(axes[1, 0], eobs, ai, "UNet", "red", median_bias=ai_q[1])
    axes[1, 0].set_title("UNet - E-OBS")

    plot_quantile_panel(axes[1, 1], {"Raw GCM": raw_q, "GCM-BC": bc_q, "UNet": ai_q})

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = Path(GRAPHS_DIR) / "metrics" / "exp5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp5_pairwise_distribution_quantiles.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
