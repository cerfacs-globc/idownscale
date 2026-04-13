"""
Compute statistics for a dataset and plot histograms for the inpus and target channels.
Save the statistics in a JSON file and the histograms as PNG files.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys

sys.path.append(".")

import argparse
import glob
import json
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from iriscc.settings import CONFIG


def update_statistics(sum: float, square_sum: float, n_total: int, min: float, max: float, x: np.ndarray) -> Tuple[float, float, int, float, float]:
    """
    Compute and update sample statistics including sum, squared sum, total count,
    minimum, and maximum values for a given array, ignoring NaN values.
    """
    x = x[~np.isnan(x)]  # Remove NaN values
    sum += np.sum(x)  # Update sum
    square_sum += np.sum(x**2)  # Update squared sum
    n_total += x.size  # Update total count
    if np.min(x) < min:  # Update minimum value
        min = np.min(x)
    if np.max(x) > max:  # Update maximum value
        max = np.max(x)
    return sum, square_sum, n_total, min, max


def plot_histogram(data, min: float, max: float, mean: float, std: float, var: str, title: str, save_dir: str):

    hist, edges = np.histogram(data, bins=50, range=(min, max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(10, 6))
    plt.bar(centers, hist, align="center", width=np.diff(edges), alpha=0.5, color="blue", label="Density")
    plt.axvline(mean, color="red", linestyle="--", linewidth=2, label=rf"$\mu$ = {mean:.2f}")
    plt.axvline(mean - std, color="green", linestyle="--", linewidth=2, label=rf" $\sigma$ = {std:.2f}")
    plt.axvline(mean + std, color="green", linestyle="--", linewidth=2)

    plt.xlabel(var, fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left", fontsize=14)
    plt.savefig(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics for a given dataset path")
    parser.add_argument("--exp", type=str, help="Experiment name", default="exp5")
    args = parser.parse_args()

    dataset_dir = CONFIG[args.exp]["dataset"]
    list_data = np.sort(glob.glob(str(dataset_dir / "sample*")))

    # Standardize splitting to match dataloader parity
    test_end_matches = np.where(list_data == str(dataset_dir / "sample_20150101.npz"))[0]
    test_end_limit = int(test_end_matches[0]) if len(test_end_matches) > 0 else len(list_data)
    dataset = list_data[:test_end_limit]

    channels = CONFIG[args.exp]["channels"]
    ch = len(channels)

    sum_vals = np.zeros(ch)
    square_sum = np.zeros(ch)
    n_total = np.zeros(ch)

    # Use 80% for training, 10% for validation, 10% for testing (matches dataloader)
    n_samples = len(dataset)
    train_end = int(0.8 * n_samples)
    val_end = int(0.9 * n_samples)

    y_data = {"train": [], "val": [], "test": []}

    for nb, sample in enumerate(dataset):
        data = dict(np.load(sample, allow_pickle=True))
        x, y = data["x"], data["y"]
        condition = np.isnan(y[0])
        for c, channel in enumerate(channels[:-1]):  # Exclude the last channel (target)
            x[c][condition] = np.nan
            if channel == "pr input":
                x[c][np.isnan(x[c])] = 0.0
                x[c] = np.log10(1 + x[c])
                x[c][condition] = np.nan

        # Only training statistics are used for normalization
        if nb < train_end:
            y_data["train"].append(y.flatten())
            if nb == 0:
                min_vals = np.concatenate((np.nanmin(x, axis=(1, 2)), np.nanmin(y, axis=(1, 2))))
                max_vals = np.concatenate((np.nanmax(x, axis=(1, 2)), np.nanmax(y, axis=(1, 2))))

            for i in range(ch):
                if i == ch - 1:
                    sum_vals[i], square_sum[i], n_total[i], min_vals[i], max_vals[i] = update_statistics(
                        sum_vals[i], square_sum[i], n_total[i], min_vals[i], max_vals[i], y[0]
                    )
                else:
                    sum_vals[i], square_sum[i], n_total[i], min_vals[i], max_vals[i] = update_statistics(
                        sum_vals[i], square_sum[i], n_total[i], min_vals[i], max_vals[i], x[i]
                    )
        elif nb < val_end:
            y_data["val"].append(y.flatten())
        else:
            y_data["test"].append(y.flatten())

    mean = sum_vals / n_total
    std = np.sqrt((square_sum / n_total) - (mean**2))

    stats = {}
    for i, chanel in enumerate(channels):
        stats[chanel] = {"mean": mean[i], "std": std[i], "min": min_vals[i].astype(np.float64), "max": max_vals[i].astype(np.float64)}

    with open(dataset_dir / "statistics.json", "w") as f:
        json.dump(stats, f)

    for type, data in y_data.items():
        if not data:
            continue
        data = np.concatenate(data)
        plot_histogram(
            data,
            np.nanmin(data),
            np.nanmax(data),
            np.nanmean(data),
            np.nanstd(data),
            CONFIG[args.exp]["target_vars"][0],
            f"y {type} dataset histogram",
            dataset_dir / f"hist_y_{type}.png",
        )
