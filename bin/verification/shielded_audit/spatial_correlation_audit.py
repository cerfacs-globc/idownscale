import sys
import os
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

# Force project root inclusion
PROJECT_ROOT = Path(__file__).parents[3].resolve()
sys.path.append(str(PROJECT_ROOT))

from iriscc.datautils import (Data, 
                              interpolation_target_grid, 
                              crop_domain_from_ds, 
                              standardize_dims_and_coords,
                              standardize_longitudes)

OROG_FILE = PROJECT_ROOT / 'rawdata/eobs/elevation_ens_025deg_reg_v29_0e.nc'
BASELINE_FILE = PROJECT_ROOT / 'data/audit/baseline_zoeleila_bc.npz'
FRANCE_DOMAIN = [-6., 12., 40., 52.]

def calculate_correlation(data_2d, topo_2d):
    mask = ~np.isnan(data_2d) & ~np.isnan(topo_2d)
    if not np.any(mask): return 0
    return np.corrcoef(data_2d[mask].flatten(), topo_2d[mask].flatten())[0, 1]

print("--- SHIELDED SPATIAL CORRELATION AUDIT ---")

# 1. Load Topography and Standardize ( Ground Truth geographical anchor)
ds_orog = xr.open_dataset(OROG_FILE)
ds_orog = standardize_dims_and_coords(ds_orog)
ds_orog = standardize_longitudes(ds_orog)
ds_orog_france = crop_domain_from_ds(ds_orog, FRANCE_DOMAIN)

# 2. Load Archival Baseline
baseline = np.load(BASELINE_FILE)
era5_base = baseline['era5_hist'][0] # 8x13 France domain

# 3. Build reference grid for 8x13 GCM regridding
get_data = Data(domain=FRANCE_DOMAIN)
date = pd.Timestamp('1979-01-01')
ds_gcm = get_data.get_gcm_dataset('tas', date)
ds_gcm_france = crop_domain_from_ds(ds_gcm, FRANCE_DOMAIN)

# Regrid Topo to 8x13
regrid_topo = interpolation_target_grid(ds_orog_france, ds_gcm_france, method="bilinear")
h_8x13 = regrid_topo.elevation.values

# 4. Correlation Analysis
corr_base = calculate_correlation(era5_base, h_8x13)
corr_base_flipped_topo = calculate_correlation(era5_base, h_8x13[::-1, :])

print(f"\n[Method 2] Correlation with Native Topography (South-to-North):")
print(f"  Archival Baseline vs Native Topo: {corr_base:.4f}")
print(f"  Archival Baseline vs Flipped Topo: {corr_base_flipped_topo:.4f}")

if abs(corr_base) > abs(corr_base_flipped_topo):
    print("\n  -> ARCHIVE AUDIT: Baseline orientation matches NATIVE (Ascending).")
    print("  -> Logic: Correlation with native geography is stronger.")
else:
    print("\n  -> ARCHIVE AUDIT: Baseline orientation matches FLIPPED (Descending).")
    print("  -> Logic: Correlation with flipped geography is stronger.")
