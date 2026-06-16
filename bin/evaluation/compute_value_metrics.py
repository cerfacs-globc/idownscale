"""
Evaluation of downscaling models using COST VALUE framework.

Compares predictions, ground truth (ERA5), and baseline.
"""

import sys

# Add root to path for imports
sys.path.append(".")

import argparse
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from iriscc.runtime_paths import require_match, resolve_runtime_sample_dir, resolve_sample_file
from iriscc.settings import (CONFIG, PREDICTION_DIR,
                             DATES_BC_TEST_HIST, METRICS_DIR,
                             get_metrics_test_name)
from iriscc.value_metrics import get_spatial_metrics


def _update_lag1_sums(prev, curr, sums):
    valid = np.isfinite(prev) & np.isfinite(curr)
    if not np.any(valid):
        return
    prev_valid = np.where(valid, prev, 0.0)
    curr_valid = np.where(valid, curr, 0.0)
    sums["count"] += valid
    sums["sum_prev"] += prev_valid
    sums["sum_curr"] += curr_valid
    sums["sum_prev2"] += prev_valid * prev_valid
    sums["sum_curr2"] += curr_valid * curr_valid
    sums["sum_cross"] += prev_valid * curr_valid


def _mean_lag1_corr(sums):
    count = sums["count"]
    numerator = count * sums["sum_cross"] - sums["sum_prev"] * sums["sum_curr"]
    denom_left = count * sums["sum_prev2"] - sums["sum_prev"] ** 2
    denom_right = count * sums["sum_curr2"] - sums["sum_curr"] ** 2
    denominator = np.sqrt(denom_left * denom_right)
    valid = (count > 1) & (denominator > 0)
    corr = np.full(count.shape, np.nan, dtype=np.float64)
    corr[valid] = numerator[valid] / denominator[valid]
    return float(np.nanmean(corr))


def _empty_lag1_sums(shape):
    return {
        "count": np.zeros(shape, dtype=np.float64),
        "sum_prev": np.zeros(shape, dtype=np.float64),
        "sum_curr": np.zeros(shape, dtype=np.float64),
        "sum_prev2": np.zeros(shape, dtype=np.float64),
        "sum_curr2": np.zeros(shape, dtype=np.float64),
        "sum_cross": np.zeros(shape, dtype=np.float64),
    }


def _std_from_sums(count, total, total_sq):
    if count <= 1:
        return np.nan
    mean = total / count
    variance = max(total_sq / count - mean * mean, 0.0)
    return float(np.sqrt(variance))

def main():
    parser = argparse.ArgumentParser(description="Compute VALUE validation metrics")
    parser.add_argument("--exp", type=str, default="exp5", help="Experiment name")
    parser.add_argument("--test-name", type=str, default="unet_all", help="Test name")
    parser.add_argument("--simu", type=str, default="gcm", help="Simulation source (gcm/rcm)")
    parser.add_argument("--simu-test", type=str, default="gcm_bc", help="Simulation test variant")
    parser.add_argument("--startdate", type=str, default=DATES_BC_TEST_HIST[0].strftime("%Y%m%d"), help="Historical validation start date")
    parser.add_argument("--enddate", type=str, default=DATES_BC_TEST_HIST[-1].strftime("%Y%m%d"), help="Historical validation end date")
    args = parser.parse_args()

    # 1. Setup paths
    metric_dir = METRICS_DIR / args.exp
    metric_dir.mkdir(parents=True, exist_ok=True)

    # 2. Identify historical test files
    # Note: These are expected in PREDICTION_DIR for the configured validation window.
    period = "historical" if pd.Timestamp(args.enddate) <= DATES_BC_TEST_HIST[-1] else CONFIG[args.exp].get("ssp", "ssp585")
    pred_files = require_match(
        PREDICTION_DIR,
        f"tas*{period}*{args.startdate}_{args.enddate}_{args.exp}_{args.test_name}_{args.simu_test}.nc",
        "prediction file",
        allow_multiple=True,
    )

    if len(pred_files) == 1:
        ds_pred = xr.open_dataset(pred_files[0])
    else:
        ds_pred = xr.concat([xr.open_dataset(path) for path in pred_files], dim="time")

    # 3. Load Target Data (ERA5)
    # The ERA5 target data is saved in the sample files or we can load it from raw if easier.
    # However, 'compute_test_metrics_day.py' loads it from the BC test dataset samples.
    # We'll do the same for consistency.
    sample_dir = resolve_runtime_sample_dir(args.exp, args.test_name, simu_test=args.simu_test)

    # 3. Load Target Data (ERA5)
    # We only load samples that match the prediction time range
    pred_dates = pd.to_datetime(ds_pred.time.values)

    rng = np.random.default_rng(0)
    max_distribution_samples = 2_000_000
    samples_per_date = max(1, int(np.ceil(max_distribution_samples / max(len(pred_dates), 1))))
    obs_distribution_samples = []
    pred_distribution_samples = []
    obs_count = pred_count = 0
    obs_sum = pred_sum = 0.0
    obs_sum_sq = pred_sum_sq = 0.0
    obs_spatial_sum = pred_spatial_sum = None
    spatial_count = None
    obs_lag1_sums = pred_lag1_sums = None
    prev_obs = prev_pred = None
    matched_dates = 0

    print(f"Loading samples and predictions for {len(pred_dates)} dates...", flush=True)
    for index, date in enumerate(tqdm(pred_dates)):
        date_str = date.strftime("%Y%m%d")
        sample_path = resolve_sample_file(sample_dir, date_str)
        data = np.load(sample_path)
        # Get target from sample
        y = np.squeeze(data["y"]).astype(np.float64) # Squeeze to handle (64,64) or (1,64,64)

        # Get prediction from netcdf for the same date
        y_hat = np.asarray(ds_pred.tas.isel(time=index).values, dtype=np.float64)
        valid = np.isfinite(y) & np.isfinite(y_hat)
        if not np.any(valid):
            continue

        if obs_spatial_sum is None:
            obs_spatial_sum = np.zeros_like(y, dtype=np.float64)
            pred_spatial_sum = np.zeros_like(y, dtype=np.float64)
            spatial_count = np.zeros_like(y, dtype=np.float64)
            obs_lag1_sums = _empty_lag1_sums(y.shape)
            pred_lag1_sums = _empty_lag1_sums(y.shape)

        obs_values = y[valid]
        pred_values = y_hat[valid]
        obs_count += obs_values.size
        pred_count += pred_values.size
        obs_sum += float(obs_values.sum())
        pred_sum += float(pred_values.sum())
        obs_sum_sq += float(np.square(obs_values).sum())
        pred_sum_sq += float(np.square(pred_values).sum())

        obs_spatial_sum[valid] += obs_values
        pred_spatial_sum[valid] += pred_values
        spatial_count[valid] += 1

        if prev_obs is not None:
            _update_lag1_sums(prev_obs, y, obs_lag1_sums)
            _update_lag1_sums(prev_pred, y_hat, pred_lag1_sums)
        prev_obs = np.where(valid, y, np.nan)
        prev_pred = np.where(valid, y_hat, np.nan)

        sample_count = min(samples_per_date, obs_values.size)
        sample_indices = rng.choice(obs_values.size, size=sample_count, replace=False)
        obs_distribution_samples.append(obs_values[sample_indices].astype(np.float32))
        pred_distribution_samples.append(pred_values[sample_indices].astype(np.float32))
        matched_dates += 1

    ds_pred.close()

    if matched_dates == 0:
        print("Error: No matching samples found for the prediction period.")
        sys.exit(1)

    # 4. Compute Metrics
    print("Computing metrics...", flush=True)

    # Marginal
    obs_sample = np.concatenate(obs_distribution_samples)
    pred_sample = np.concatenate(pred_distribution_samples)
    obs_mean = obs_sum / obs_count
    pred_mean = pred_sum / pred_count
    obs_std = _std_from_sums(obs_count, obs_sum, obs_sum_sq)
    pred_std = _std_from_sums(pred_count, pred_sum, pred_sum_sq)
    marginal = {
        "bias": pred_mean - obs_mean,
        "std_ratio": pred_std / obs_std,
        "q5_bias": np.nanquantile(pred_sample, 0.05) - np.nanquantile(obs_sample, 0.05),
        "q50_bias": np.nanquantile(pred_sample, 0.50) - np.nanquantile(obs_sample, 0.50),
        "q95_bias": np.nanquantile(pred_sample, 0.95) - np.nanquantile(obs_sample, 0.95),
        "wasserstein": wasserstein_distance(obs_sample, pred_sample),
        "distribution_sample_size": len(obs_sample),
        "matched_days": matched_dates,
    }

    avg_temporal = {
        "autocorr_obs_mean": _mean_lag1_corr(obs_lag1_sums),
        "autocorr_pred_mean": _mean_lag1_corr(pred_lag1_sums),
    }
    avg_temporal["autocorr_bias_mean"] = avg_temporal["autocorr_pred_mean"] - avg_temporal["autocorr_obs_mean"]

    # Spatial (on mean maps)
    with np.errstate(invalid="ignore", divide="ignore"):
        obs_mean_map = obs_spatial_sum / spatial_count
        pred_mean_map = pred_spatial_sum / spatial_count
    spatial = get_spatial_metrics(obs_mean_map, pred_mean_map)

    # 5. Summarize
    all_metrics = {**marginal, **avg_temporal, **spatial}

    df = pd.DataFrame([all_metrics])
    metrics_test_name = get_metrics_test_name(args.test_name, args.simu_test)
    output_path = metric_dir / f"value_metrics_{args.exp}_{metrics_test_name}.csv"
    df.to_csv(output_path, index=False)

    print("\nVALUE Summary Table:", flush=True)
    try:
        print(df.to_markdown(index=False), flush=True)
    except ImportError:
        print(df.to_string(index=False), flush=True)

    print(f"\nResults saved to {output_path}", flush=True)

if __name__ == "__main__":
    main()
