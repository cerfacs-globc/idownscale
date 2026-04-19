'''
Spatial Bias Audit: Point Enumeration
Quantifies the number of grid points with large temperature discrepancies.
(Restricted to idownscale_rerun, idownscale_exp5, and idownscale_output)

date : 19/04/2026
author : Antigravity (AI Assistant)
'''
import sys
from pathlib import Path

# Force the project root into the path
PROJECT_ROOT = Path(__file__).parents[2].resolve()
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import xarray as xr
from iriscc.datautils import Data, interpolation_target_grid, reformat_as_target
from iriscc.settings import DATASET_EXP5_30Y_DIR, CONFIG

def audit_spatial_bias(date_str):
    date = pd.Timestamp(date_str)
    exp = 'exp5'
    
    # 1. Load Baseline (Archive)
    baseline_path = DATASET_EXP5_30Y_DIR / f"sample_{date.strftime('%Y%m%d')}.npz"
    baseline = np.load(baseline_path)
    X_base = baseline['x'][1] # Temperature channel
    
    # 2. Setup Current Data (Raw/Uncorrected Rerun)
    get_data = Data(domain=CONFIG[exp]['domain'])
    ds_era5 = get_data.get_era5_dataset('tas', date)
    ds_gcm = get_data.get_gcm_dataset('tas', date, CONFIG[exp]['ssp'])
    
    # Run standard production regridding
    ds_era5_to_gcm = interpolation_target_grid(ds_era5, ds_target=ds_gcm, method="conservative_normed")
    ds_rerun = reformat_as_target(ds_era5_to_gcm, 
                                target_file=CONFIG[exp]['target_file'], 
                                method='conservative_normed', 
                                domain=CONFIG[exp]['domain'], 
                                crop_target=True, mask=True)
    X_rerun = ds_rerun['tas'].values
    
    # 3. Calculate Differences
    diff = np.abs(X_rerun - X_base)
    mask = ~np.isnan(X_rerun) & ~np.isnan(X_base)
    valid_points = np.sum(mask)
    
    # Threshold Audit
    bias_2k = np.sum(diff[mask] > 2.0)
    bias_5k = np.sum(diff[mask] > 5.0)
    bias_10k = np.sum(diff[mask] > 10.0)
    
    print(f"--- Spatial Bias Audit ({date_str}) ---")
    print(f"Total Valid Grid Points: {valid_points}")
    print(f"\nDiscrepancy Distribution:")
    print(f"Points >  2.0 K: {bias_2k:5d} ({100*bias_2k/valid_points:4.1f}%) [Significant]")
    print(f"Points >  5.0 K: {bias_5k:5d} ({100*bias_5k/valid_points:4.1f}%) [Extreme]")
    print(f"Points > 10.0 K: {bias_10k:5d} ({100*bias_10k/valid_points:4.1f}%) [Alpine Peaks]")
    
    print(f"\nMax Discrepancy: {np.max(diff[mask]):.2f} K")
    print(f"Mean Absolute Error: {np.mean(diff[mask]):.2f} K")

if __name__ == "__main__":
    audit_spatial_bias("1980-01-01")
