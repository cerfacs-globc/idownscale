"""
Evaluation of downscaling models using COST VALUE framework.

Compares predictions, ground truth (ERA5), and baseline.
"""

import sys
from pathlib import Path

# Add root to path for imports
sys.path.append('.')

import argparse
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from iriscc.settings import (CONFIG, PREDICTION_DIR, DATASET_BC_DIR,
                             DATES_BC_TEST_HIST, METRICS_DIR,
                             get_evaluation_sample_dir, get_metrics_test_name)
from iriscc.value_metrics import (get_marginal_metrics, get_temporal_metrics, 
                                  get_spatial_metrics, get_spell_length)

def main():
    parser = argparse.ArgumentParser(description="Compute VALUE validation metrics")
    parser.add_argument('--exp', type=str, default='exp5', help='Experiment name')
    parser.add_argument('--test-name', type=str, default='unet_all', help='Test name')
    parser.add_argument('--simu', type=str, default='gcm', help='Simulation source (gcm/rcm)')
    parser.add_argument('--simu-test', type=str, default='gcm_bc', help='Simulation test variant')
    parser.add_argument('--startdate', type=str, default=DATES_BC_TEST_HIST[0].strftime('%Y%m%d'), help='Historical validation start date')
    parser.add_argument('--enddate', type=str, default=DATES_BC_TEST_HIST[-1].strftime('%Y%m%d'), help='Historical validation end date')
    args = parser.parse_args()

    # 1. Setup paths
    metric_dir = METRICS_DIR / args.exp
    metric_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Identify historical test files
    # Note: These are expected in PREDICTION_DIR for the configured validation window.
    period = 'historical' if pd.Timestamp(args.enddate) <= DATES_BC_TEST_HIST[-1] else CONFIG[args.exp].get('ssp', 'ssp585')
    pred_files = list(
        PREDICTION_DIR.glob(
            f'tas*{period}*{args.startdate}_{args.enddate}_{args.exp}_{args.test_name}_{args.simu_test}.nc'
        )
    )
    if not pred_files:
        print(f"Error: No prediction files found for {args.exp}/{args.test_name} in historical period.")
        sys.exit(1)
        
    if len(pred_files) == 1:
        ds_pred = xr.open_dataset(pred_files[0])
    else:
        ds_pred = xr.concat([xr.open_dataset(path) for path in pred_files], dim="time")
    
    # 3. Load Target Data (ERA5)
    # The ERA5 target data is saved in the sample files or we can load it from raw if easier.
    # However, 'compute_test_metrics_day.py' loads it from the BC test dataset samples.
    # We'll do the same for consistency.
    sample_dir = get_evaluation_sample_dir(args.exp, args.test_name, args.simu_test) or DATASET_BC_DIR / f'dataset_{args.exp}_test_{args.simu_test}'
    
    dates = []
    obs_list = []
    pred_list = []
    
    # 3. Load Target Data (ERA5)
    # We only load samples that match the prediction time range
    pred_dates = pd.to_datetime(ds_pred.time.values)
    
    dates = []
    obs_list = []
    pred_list = []
    
    print(f"Loading samples and predictions for {len(pred_dates)} dates...", flush=True)
    for date in tqdm(pred_dates):
        date_str = date.strftime('%Y%m%d')
        sample_path = sample_dir / f'sample_{date_str}.npz'
        
        if sample_path.exists():
            data = np.load(sample_path)
            # Get target from sample
            y = np.squeeze(data['y']) # Squeeze to handle (64,64) or (1,64,64)
            condition = np.isnan(y)
            obs_list.append(y)
            
            # Get prediction from netcdf for the same date
            y_hat = ds_pred.tas.sel(time=date, method='nearest').values
            y_hat = np.array(y_hat, copy=True)
            y_hat[condition] = np.nan
            pred_list.append(y_hat)
            dates.append(date)

    if not obs_list:
        print("Error: No matching samples found for the prediction period.")
        sys.exit(1)

    obs = np.stack(obs_list)
    pred = np.stack(pred_list)
    
    # 4. Compute Metrics
    print("Computing metrics...", flush=True)
    
    # Marginal
    marginal = get_marginal_metrics(obs, pred)
    
    # Temporal (at each pixel, then mean)
    # For lag-1 autocorr, we need the full timeseries at each point
    obs_reshaped = obs.reshape(obs.shape[0], -1)
    pred_reshaped = pred.reshape(pred.shape[0], -1)
    
    temp_metrics_list = [
        get_temporal_metrics(obs_reshaped[:, i], pred_reshaped[:, i])
        for i in range(obs_reshaped.shape[1])
        if not np.all(np.isnan(obs_reshaped[:, i]))
    ]
    
    avg_temporal = {
        'autocorr_obs_mean': np.nanmean([m['autocorr_obs'] for m in temp_metrics_list]),
        'autocorr_pred_mean': np.nanmean([m['autocorr_pred'] for m in temp_metrics_list]),
        'autocorr_bias_mean': np.nanmean([m['autocorr_error'] for m in temp_metrics_list])
    }
    
    # Spatial (on mean maps)
    obs_mean_map = np.nanmean(obs, axis=0)
    pred_mean_map = np.nanmean(pred, axis=0)
    spatial = get_spatial_metrics(obs_mean_map, pred_mean_map)
    
    # 5. Summarize
    all_metrics = {**marginal, **avg_temporal, **spatial}
    
    df = pd.DataFrame([all_metrics])
    metrics_test_name = get_metrics_test_name(args.test_name, args.simu_test)
    output_path = metric_dir / f'value_metrics_{args.exp}_{metrics_test_name}.csv'
    df.to_csv(output_path, index=False)
    
    print("\nVALUE Summary Table:", flush=True)
    try:
        print(df.to_markdown(index=False), flush=True)
    except ImportError:
        print(df.to_string(index=False), flush=True)
    
    print(f"\nResults saved to {output_path}", flush=True)

if __name__ == '__main__':
    main()
