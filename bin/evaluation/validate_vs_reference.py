"""
Validation script to compare current run results with original Reference Experiment 5.

date : 09/04/2026
author : Antigravity (AI Assistant)
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
OUR_PRED_DIR = Path('/scratch/globc/page/idownscale_garcia_clean/prediction/')
REF_PRED_DIR = Path('/scratch/globc/garcia/idownscale/prediction/')
OUR_METRICS_DIR = Path('/scratch/globc/page/idownscale_garcia_clean/metrics/exp5/')
REF_METRICS_DIR = Path('/scratch/globc/garcia/idownscale/metrics/exp5/')
GRAPH_DIR = Path('/scratch/globc/page/idownscale_garcia_clean/graph/validation/')

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

def compare_predictions():
    print("--- Comparing Historical Predictions (2000-2014) ---")
    
    # Filenames
    our_file = OUR_PRED_DIR / 'tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_gcm_bc.nc'
    ref_file = REF_PRED_DIR / 'tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101_20141231_exp5_unet_all_gcm_bc.nc'
    
    if not our_file.exists():
        print(f"WAIT: Our prediction file not found yet: {our_file}")
        return
    if not ref_file.exists():
        print(f"ERROR: Reference prediction file not found: {ref_file}")
        return
        
    ds_our = xr.open_dataset(our_file)
    ds_ref = xr.open_dataset(ref_file)
    
    # Align dates (ref has 1980-2014, our has 2000-2014)
    ds_ref_subset = ds_ref.sel(time=slice('2000-01-01', '2014-12-31'))
    
    # Calculate Spatial Means
    our_mean = ds_our.tas.mean(dim=['lat', 'lon']).values
    ref_mean = ds_ref_subset.tas.mean(dim=['lat', 'lon']).values
    
    # Calculate Differences
    diff = our_mean - ref_mean
    rmse = np.sqrt(np.mean(diff**2))
    bias = np.mean(diff)
    
    print(f"Comparison Results (Our vs Ref):")
    print(f"  Mean Bias: {bias:.4f} K")
    print(f"  RMSE: {rmse:.4f} K")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(ds_our.time, our_mean, label='Current Run (Clean Room)', alpha=0.7)
    plt.plot(ds_our.time, ref_mean, label='Original Reference Exp5', linestyle='--', alpha=0.7)
    plt.title('Daily Spatial Mean Temperature Comparison (2000-2014)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / 'prediction_comparison_timeseries.png')
    print(f"Saved comparison plot to {GRAPH_DIR / 'prediction_comparison_timeseries.png'}")

    # Check for cold bias resolution
    # In previous discussions, a 26K cold bias was mentioned.
    # If our mean is ~280-290K and ref is also there, we are good.
    print(f"\nAverage Temperature (Our): {np.mean(our_mean):.2f} K")
    print(f"Average Temperature (Ref): {np.mean(ref_mean):.2f} K")
    
    if np.mean(our_mean) < 250:
        print("WARNING: Cold bias detected in current run!")
    else:
        print("SUCCESS: Temperature levels look physically consistent.")

if __name__ == "__main__":
    compare_predictions()
