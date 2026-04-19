'''
Topographic Correlation Audit
Quantifies significant biases and their relationship to high elevation.
Compares stabilized rerun against authentic archival baseline.

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
from bin.preprocessing.build_dataset import DatasetBuilder

def audit_topo_bias(date_str):
    date = pd.Timestamp(date_str)
    
    # 1. Load Authentic Archival Baseline
    baseline_dir = Path("/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/")
    baseline_path = baseline_dir / f"sample_{date.strftime('%Y%m%d')}.npz"
    baseline = np.load(baseline_path)
    X_base = baseline['x'][1]  # Temperature
    Elev = baseline['x'][0]    # Elevation channel
    
    # 2. Regenerate Stablized Rerun (with dynamic lapse correction)
    get_builder = DatasetBuilder(exp='exp5')
    X_new_full = get_builder.input_data(date)
    X_new = X_new_full[1] # Temperature
    
    # 3. Calculate Differences
    diff = np.abs(X_new - X_base)
    mask = ~np.isnan(X_new) & ~np.isnan(X_base)
    valid_points = np.sum(mask)
    
    print(f"--- Topographic Correlation Audit ({date_str}) ---")
    print(f"Total Valid Grid Points: {valid_points}")
    
    # Thresholding
    thresholds = [0.1, 0.5, 1.0, 2.0]
    for t in thresholds:
        high_bias_mask = mask & (diff > t)
        count = np.sum(high_bias_mask)
        if count > 0:
            avg_elev = np.mean(Elev[high_bias_mask])
            print(f"Points > {t:3.1f} K: {count:5d} ({100*count/valid_points:4.1f}%) | Avg Elevation: {avg_elev:6.1f}m")
        else:
            print(f"Points > {t:3.1f} K:     0 ( 0.0%)")

    # Domain Stats for comparison
    print(f"\nDomain Avg Elevation: {np.mean(Elev[mask]):6.1f} m")
    print(f"Max Elevation:         {np.max(Elev[mask]):6.1f} m")
    print(f"Max Bias:              {np.max(diff[mask]):6.2f} K")

if __name__ == "__main__":
    audit_topo_bias("1980-01-01")
