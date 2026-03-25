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
                             DATASET_DIR, METRICS_DIR)
from iriscc.value_metrics import (get_marginal_metrics, get_temporal_metrics, 
                                  get_spatial_metrics, get_spell_length)

def main():
    parser = argparse.ArgumentParser(description="Compute VALUE validation metrics")
    parser.add_argument('--exp', type=str, default='exp5', help='Experiment name')
    parser.add_argument('--test-name', type=str, default='unet_all', help='Test name')
    parser.add_argument('--simu', type=str, default='gcm', help='Simulation source (gcm/rcm)')
    parser.add_argument('--simu-test', type=str, default='gcm_bc', help='Simulation test variant')
    args = parser.parse_args()

    # 1. Setup paths
    metric_dir = METRICS_DIR / args.exp
    metric_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Identify historical test files
    # Note: These are expected in PREDICTION_DIR for the and test period (2000-2014)
    # We use the naming pattern: tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_exp5_unet_all_gcm_bc.nc
    pred_files = list(PREDICTION_DIR.glob(f'tas*historical*{args.exp}_{args.test_name}_{args.simu_test}.nc'))
    if not pred_files:
        print(f"Error: No prediction files found for {args.exp}/{args.test_name} in historical period.")
        sys.exit(1)
        
    ds_pred = xr.open_mfdataset(pred_files)
    
    # 3. Load Target Data (ERA5)
    # The ERA5 target data is saved in the sample files or we can load it from raw if easier.
    # However, 'compute_test_metrics_day.py' loads it from the BC test dataset samples.
    # We'll do the same for consistency.
    sample_dir = DATASET_BC_DIR / f'dataset_{args.exp}_test_{args.simu_test}'
    samples = sorted(list(sample_dir.glob('sample_*.npz')))
    
    dates = []
    obs_list = []
    pred_list = []
    
    print(f"Loading {len(samples)} samples and predictions...", flush=True)
    for sample_path in tqdm(samples):
        data = np.load(sample_path)
        date_str = sample_path.stem.split('_')[1]
        date = pd.to_datetime(date_str)
        
        # Get target from sample
        y = data['y'][0] # Target is typically the first channel
        obs_list.append(y)
        
        # Get prediction from netcdf for the same date
        y_hat = ds_pred.tas.sel(time=date, method='nearest').values
        pred_list.append(y_hat)
        dates.append(date)

    obs = np.stack(obs_list)
    pred = np.stack(pred_list)
    
    # 4. Compute Metrics
    print("Computing metrics...", flush=True)
    
    # Marginal
    marginal = get_marginal_metrics(obs, pred)
    
    # Temporal (at each pixel, then mean)
    # For lag-1 autocorr, we need the full timeseries at each point
    n_pixels = obs.shape[1] * obs.shape[2]
    obs_reshaped = obs.reshape(obs.shape[0], -1)
    pred_reshaped = pred.reshape(pred.shape[0], -1)
    
    temp_metrics_list = []
    for i in range(obs_reshaped.shape[1]):
        if not np.all(np.isnan(obs_reshaped[:, i])):
            temp_metrics_list.append(get_temporal_metrics(obs_reshaped[:, i], pred_reshaped[:, i]))
    
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
    output_path = metric_dir / f'value_metrics_{args.exp}_{args.test_name}.csv'
    df.to_csv(output_path, index=False)
    
    print("\nVALUE Summary Table:")
    print(df.to_markdown())
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()
