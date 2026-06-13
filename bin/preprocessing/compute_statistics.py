"""
Compute statistics for a dataset and plot histograms for the inpus and target channels.

Save the statistics in a JSON file and the histograms as PNG files.

date : 16/07/2025
author : Zoé GARCIA
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np

from iriscc.provenance import build_prov_bundle, print_resolved_context, utc_now_iso, write_provjson
from iriscc.settings import CONFIG, get_train_split_dates

def update_statistics(current_sum: float,
                      square_sum: float,
                      n_total: int,
                      min_val: float,
                      max_val: float,
                      x: np.ndarray) -> tuple[float, float, int, float, float]:
    """Compute and update sample statistics including sum, squared sum, total count, minimum, and maximum values for a given array, ignoring NaN values."""
    x = x[~np.isnan(x)]  # Remove NaN values
    current_sum += np.sum(x)  # Update sum
    square_sum += np.sum(x**2)  # Update squared sum
    n_total += x.size  # Update total count
    min_val = min(min_val, np.min(x))
    max_val = max(max_val, np.max(x))
    return current_sum, square_sum, n_total, min_val, max_val


def plot_histogram(data,
                   min_val:float,
                   max_val:float,
                   mean:float,
                   std:float,
                   var:str,
                   title:str,
                   save_dir:str):

    hist, edges = np.histogram(data, bins=50, range=(min_val, max_val), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(10, 6))
    plt.bar(centers, hist, align="center", width=np.diff(edges), alpha=0.5, color="blue", label="Density")
    plt.axvline(mean, color="red", linestyle="--", linewidth=2, label=fr"$\mu$ = {mean:.2f}")
    plt.axvline(mean - std, color="green", linestyle="--", linewidth=2, label=fr" $\sigma$ = {std:.2f}")
    plt.axvline(mean + std, color="green", linestyle="--", linewidth=2)

    plt.xlabel(var, fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper left", fontsize=14)
    plt.savefig(save_dir)


if __name__=="__main__":
    start_time = utc_now_iso()
    parser = argparse.ArgumentParser(description="Compute statistics for a given dataset path")
    parser.add_argument("--exp", type=str, help="Experiment name", default="exp6")
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset directory", default=None)
    parser.add_argument("--train-start-date", type=str, default=None, help="Training split start date YYYYMMDD.")
    parser.add_argument("--val-start-date", type=str, default=None, help="Validation split start date YYYYMMDD.")
    parser.add_argument("--test-start-date", type=str, default=None, help="Test split start date YYYYMMDD.")
    parser.add_argument("--force", action="store_true", help="Force reconnection")
    args = parser.parse_args()

    split_dates = get_train_split_dates(args.exp)
    args.train_start_date = args.train_start_date or split_dates[0]
    args.val_start_date = args.val_start_date or split_dates[1]
    args.test_start_date = args.test_start_date or split_dates[2]

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).resolve()
    else:
        dataset_dir = CONFIG[args.exp]["dataset"].resolve()
    resolved_settings = {
        "exp": args.exp,
        "dataset_dir": dataset_dir,
        "channels": CONFIG[args.exp]["channels"],
        "train_start_date": args.train_start_date,
        "val_start_date": args.val_start_date,
        "test_start_date": args.test_start_date,
    }
    print_resolved_context(
        script_name="compute_statistics.py",
        parameters=vars(args),
        settings=resolved_settings,
        inputs={"dataset_dir": dataset_dir},
        outputs={"statistics_json": dataset_dir / "statistics.json"},
    )
    stats_file = dataset_dir / "statistics.json"
    if stats_file.exists() and not args.force:
        print(f"Skipping statistics computation: {stats_file} already exists.", flush=True)
        sys.exit(0)

    dataset = np.sort([str(p.resolve()) for p in dataset_dir.glob("sample*")])
    channels = CONFIG[args.exp]["channels"]
    ch = len(channels)
    total_sum = np.zeros(ch)
    square_sum = np.zeros(ch)
    n_total = np.zeros(ch)

    # Other way to split the dataset
    #nb = len(dataset)
    #train_end = int(0.6 * nb)
    #val_end = train_end + int(0.2 * nb)

    train_start = np.where(dataset == str((dataset_dir / f"sample_{args.train_start_date}.npz").resolve()))[0][0]
    val_start = np.where(dataset == str((dataset_dir / f"sample_{args.val_start_date}.npz").resolve()))[0][0]
    test_start = np.where(dataset == str((dataset_dir / f"sample_{args.test_start_date}.npz").resolve()))[0][0]

    y_data = {"train" : [],
                 "val" : [],
                 "test" : []}

    for nb, sample in enumerate(dataset):
        print(sample)

        data = dict(np.load(sample, allow_pickle=True))
        x = data["x"]
        y = data.get("y")

        if y is not None:
            condition = np.isnan(y[0])
        else:
            # Fallback condition if no y is present (e.g. inference samples)
            condition = np.isnan(x[0])

        for c, channel in enumerate(channels[:-1]):  # Exclude the last channel (target)
            x[c][condition] = np.nan
            if channel == "pr input":
                x[c][np.isnan(x[c])] = 0.
                x[c] = np.log10(1 + x[c])  # Apply log transformation to precipitation input
                x[c][condition] = np.nan
                print(np.nanmax(x[c]))

        # Only training statistics are used for normalization
        if nb in range(train_start, val_start-1):
            y_data["train"].append(y.flatten())
            if nb == train_start:
                min_vals, max_vals = np.nanmin(x, axis=(1, 2)), np.nanmax(x, axis=(1, 2))
                min_vals = np.concatenate((min_vals, np.nanmin(y, axis=(1, 2))))
                max_vals = np.concatenate((max_vals, np.nanmax(y, axis=(1, 2))))

            for i in range(ch):
                if i == ch-1:
                    total_sum[i], square_sum[i], n_total[i], min_vals[i], max_vals[i] = update_statistics(total_sum[i],
                                                                        square_sum[i],
                                                                        n_total[i],
                                                                        min_vals[i],
                                                                        max_vals[i],
                                                                        y[0])
                else:
                    total_sum[i], square_sum[i], n_total[i], min_vals[i], max_vals[i] = update_statistics(total_sum[i],
                                                                        square_sum[i],
                                                                        n_total[i],
                                                                        min_vals[i],
                                                                        max_vals[i],
                                                                        x[i])



        # Validation and test data histograms
        elif nb in range(val_start, test_start-1):
            if y is not None:
                y_data["val"].append(y.flatten())
        elif nb >= test_start :
            if y is not None:
                y_data["test"].append(y.flatten())

    mean = total_sum / n_total
    std = np.sqrt((square_sum / n_total) - (mean**2))

    stats = {}
    for i, chanel in enumerate(channels):
        stats[chanel] = {"mean": mean[i],
                         "std": std[i],
                         "min": min_vals[i].astype(np.float64),
                         "max": max_vals[i].astype(np.float64)}

    with (dataset_dir / "statistics.json").open("w") as f:
        json.dump(stats, f)

    for data_type, data_list in y_data.items():
        data_arr = np.concatenate(data_list)
        plot_histogram(data_arr,
                    np.nanmin(data_arr),
                    np.nanmax(data_arr),
                    np.nanmean(data_arr),
                    np.nanstd(data_arr),
                    CONFIG[args.exp]["target_vars"][0],
                    f"y {data_type} dataset histogram",
                    dataset_dir/f"hist_y_{data_type}.png")
    prov_path = write_provjson(
        dataset_dir / "provenance_statistics.prov.json",
        build_prov_bundle(
            script_name="compute_statistics.py",
            activity_type="statistics",
            start_time=start_time,
            end_time=utc_now_iso(),
            parameters=vars(args),
            settings=resolved_settings,
            inputs={"dataset_dir": dataset_dir},
            outputs={
                "statistics_json": dataset_dir / "statistics.json",
                "hist_y_train_png": dataset_dir / "hist_y_train.png",
                "hist_y_val_png": dataset_dir / "hist_y_val.png",
                "hist_y_test_png": dataset_dir / "hist_y_test.png",
            },
            cwd=Path(__file__).resolve().parents[2],
        ),
    )
    print(f"provenance_provjson={prov_path}", flush=True)


