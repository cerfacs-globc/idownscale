#!/usr/bin/env python3
"""Compute day/month summary metrics from an existing prediction NetCDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(".")

from iriscc.settings import (
    CONFIG,
    DATES_BC_TEST_HIST,
    METRICS_DIR,
    get_evaluation_sample_dir,
    get_metrics_test_name,
    get_prediction_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--simu-test", required=True)
    parser.add_argument("--var", default=None)
    parser.add_argument("--startdate", default=DATES_BC_TEST_HIST[0].strftime("%Y%m%d"))
    parser.add_argument("--enddate", default=DATES_BC_TEST_HIST[-1].strftime("%Y%m%d"))
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--prediction-path", default=None)
    parser.add_argument("--frequency", choices=["daily", "monthly"], default="daily")
    parser.add_argument("--suffix", default="", help="Optional suffix appended before the output extension, e.g. _pp.")
    return parser.parse_args()


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    a_valid = a[valid]
    b_valid = b[valid]
    a_std = float(a_valid.std())
    b_std = float(b_valid.std())
    if a_std == 0.0 or b_std == 0.0:
        return float("nan")
    return float(np.corrcoef(a_valid, b_valid)[0, 1])


def flatten_valid(y: np.ndarray, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(y) & np.isfinite(y_hat)
    return y[valid].astype(np.float64), y_hat[valid].astype(np.float64)


def summarize(records: list[tuple[pd.Timestamp, np.ndarray, np.ndarray]]) -> dict[str, object]:
    rmse_temporal: list[float] = []
    corr_spatial: list[float] = []
    y_temporal: list[np.ndarray] = []
    y_hat_temporal: list[np.ndarray] = []
    rmse_spatial = None
    bias_spatial = None
    seasonal = {
        "summer": {"idx": [], "rmse_spatial": None, "bias_spatial": None},
        "winter": {"idx": [], "rmse_spatial": None, "bias_spatial": None},
    }

    for idx, (date, y, y_hat) in enumerate(records):
        error = y_hat - y
        error_squared = error**2
        rmse_spatial = error_squared if rmse_spatial is None else rmse_spatial + error_squared
        bias_spatial = error if bias_spatial is None else bias_spatial + error
        if date.month in (6, 7, 8):
            bucket = seasonal["summer"]
            bucket["idx"].append(idx)
            bucket["rmse_spatial"] = error_squared if bucket["rmse_spatial"] is None else bucket["rmse_spatial"] + error_squared
            bucket["bias_spatial"] = error if bucket["bias_spatial"] is None else bucket["bias_spatial"] + error
        if date.month in (1, 2, 12):
            bucket = seasonal["winter"]
            bucket["idx"].append(idx)
            bucket["rmse_spatial"] = error_squared if bucket["rmse_spatial"] is None else bucket["rmse_spatial"] + error_squared
            bucket["bias_spatial"] = error if bucket["bias_spatial"] is None else bucket["bias_spatial"] + error

        y_flat, y_hat_flat = flatten_valid(y, y_hat)
        rmse_temporal.append(float(np.sqrt(np.mean((y_hat_flat - y_flat) ** 2))))
        corr_spatial.append(pearson(y_hat_flat, y_flat))
        y_temporal.append(y_flat)
        y_hat_temporal.append(y_hat_flat)

    y_stack = np.stack(y_temporal)
    y_hat_stack = np.stack(y_hat_temporal)
    d_t = np.diff(y_stack, axis=0)
    d_t_hat = np.diff(y_hat_stack, axis=0)
    variability = np.mean(np.abs(d_t_hat), axis=0) - np.mean(np.abs(d_t), axis=0)
    corr_temporal = np.array([pearson(y_hat_stack[:, j], y_stack[:, j]) for j in range(y_stack.shape[1])])

    def seasonal_values(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = seasonal[name]["idx"]
        if not idx:
            nan_map = np.full_like(records[0][1], np.nan, dtype=np.float64)
            return np.array([np.nan]), np.array([np.nan]), nan_map, nan_map, np.array([np.nan])
        rmse_values = np.array([rmse_temporal[i] for i in idx])
        corr_values = np.array([corr_spatial[i] for i in idx])
        rmse_map = np.sqrt(seasonal[name]["rmse_spatial"] / len(idx))
        bias_map = seasonal[name]["bias_spatial"] / len(idx)
        diff_idx = [i - 1 for i in idx if i > 0]
        if diff_idx:
            var_values = np.mean(np.abs(d_t_hat[diff_idx]), axis=0) - np.mean(np.abs(d_t[diff_idx]), axis=0)
        else:
            var_values = np.array([np.nan])
        return rmse_values, corr_values, rmse_map, bias_map, var_values

    rmse_spatial = np.sqrt(rmse_spatial / len(records))
    bias_spatial = bias_spatial / len(records)
    summer_rmse, summer_corr, summer_rmse_map, summer_bias_map, summer_var = seasonal_values("summer")
    winter_rmse, winter_corr, winter_rmse_map, winter_bias_map, winter_var = seasonal_values("winter")

    mean_table = pd.DataFrame(
        {
            "rmse_temporal_mean": [np.nanmean(rmse_temporal), np.nanmean(summer_rmse), np.nanmean(winter_rmse)],
            "rmse_spatial_mean": [np.nanmean(rmse_spatial), np.nanmean(summer_rmse_map), np.nanmean(winter_rmse_map)],
            "bias_spatial_mean": [np.nanmean(bias_spatial), np.nanmean(summer_bias_map), np.nanmean(winter_bias_map)],
            "bias_spatial_std": [np.nanstd(bias_spatial), np.nanstd(summer_bias_map), np.nanstd(winter_bias_map)],
            "corr_spatial_mean": [np.nanmean(corr_spatial), np.nanmean(summer_corr), np.nanmean(winter_corr)],
            "corr_temporal_mean": [np.nanmean(corr_temporal), np.nan, np.nan],
            "variability_mean": [np.nanmean(variability), np.nanmean(summer_var), np.nanmean(winter_var)],
        },
        index=["all", "summer", "winter"],
    )
    detail = {
        "rmse_temporal": np.array(rmse_temporal),
        "rmse_spatial": rmse_spatial.flatten(),
        "bias_spatial": bias_spatial.flatten(),
        "corr_temporal": corr_temporal,
        "corr_spatial": np.array(corr_spatial),
        "variability": variability,
    }
    return {"mean": mean_table, "detail": detail}


def main() -> int:
    args = parse_args()
    var = args.var or CONFIG[args.exp]["target_vars"][0]
    sample_dir = Path(args.sample_dir) if args.sample_dir else get_evaluation_sample_dir(args.exp, args.test_name, args.simu_test)
    if sample_dir is None:
        raise ValueError("Could not resolve sample directory; pass --sample-dir explicitly.")
    prediction_path = (
        Path(args.prediction_path)
        if args.prediction_path
        else get_prediction_output_path(
            args.exp,
            args.simu_test,
            var,
            args.startdate,
            args.enddate,
            get_metrics_test_name(args.test_name, args.simu_test),
            ssp=CONFIG[args.exp].get("ssp"),
        )
    )

    with xr.open_dataset(prediction_path) as ds:
        pred_dates = pd.to_datetime(ds.time.values)
        records = []
        for date in pred_dates:
            date_str = date.strftime("%Y%m%d")
            sample = np.load(sample_dir / f"sample_{date_str}.npz")
            y = np.squeeze(sample["y"]).astype(np.float64)
            y_hat = np.asarray(ds[var].sel(time=date).values, dtype=np.float64)
            y_hat[np.isnan(y)] = np.nan
            records.append((date, y, y_hat))

    if args.frequency == "monthly":
        monthly_records = []
        for _, group in pd.DataFrame({"idx": range(len(records)), "date": [r[0] for r in records]}).groupby(pd.Grouper(key="date", freq="MS")):
            if group.empty:
                continue
            y = np.nanmean(np.stack([records[i][1] for i in group["idx"]]), axis=0)
            y_hat = np.nanmean(np.stack([records[i][2] for i in group["idx"]]), axis=0)
            monthly_records.append((pd.Timestamp(group["date"].iloc[0]), y, y_hat))
        records = monthly_records

    result = summarize(records)
    metric_dir = METRICS_DIR / args.exp / "mean_metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    metrics_test_name = get_metrics_test_name(args.test_name, args.simu_test)
    suffix = args.suffix
    detail_path = metric_dir / f"metrics_test_{args.frequency}_{args.exp}_{metrics_test_name}{suffix}.npz"
    mean_path = metric_dir / f"metrics_test_mean_{args.frequency}_{args.exp}_{metrics_test_name}{suffix}.csv"
    np.savez(detail_path, **result["detail"], dates=np.array([r[0] for r in records]))
    result["mean"].to_csv(mean_path)
    print(result["mean"])
    print(f"detail={detail_path}")
    print(f"mean={mean_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
