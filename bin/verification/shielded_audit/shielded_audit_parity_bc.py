import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

# Force project root inclusion
PROJECT_ROOT = Path(__file__).parents[3].resolve()
sys.path.append(str(PROJECT_ROOT))

FRANCE_DOMAIN = [-6., 12., 40., 52.]

from iriscc.datautils import (Data, interpolation_target_grid, crop_domain_from_ds)

def shielded_audit(date_str):
    date = pd.Timestamp(date_str)
    
    # 1. Load Archival Baseline
    baseline_path = PROJECT_ROOT / "data/audit/baseline_zoeleila_bc.npz"
    baseline = np.load(baseline_path)
    Era5_base = baseline['era5_hist'][0].astype(np.float64)
    
    # 2. Build synthesis foundation (Step: Topo Correction DISABLED)
    print(f"--- SHIELDED Phase 2 Audit (Diagnostic: No Topo Correction) ---")
    get_data = Data(domain=FRANCE_DOMAIN)
    
    # Retrieve ERA5 with Native orientation and NO topographic restoration
    ds_era5 = get_data.get_era5_dataset('tas', date, lapse_rate_correction=False)
    
    # Ibicus logic: regrid to GCM coarse grid (8x13)
    ds_gcm = get_data.get_gcm_dataset('tas', date)
    
    # Match Orientations for comparison
    # (Since Method 2 certified Baseline=Ascending, we flip our native Descending foundations)
    ds_era5 = ds_era5.reindex(lat=ds_era5.lat[::-1])
    if ds_gcm.lat.values[1] < ds_gcm.lat.values[0]:
        ds_gcm = ds_gcm.reindex(lat=ds_gcm.lat[::-1])
    
    ds_era5_to_gcm = interpolation_target_grid(ds_era5, ds_target=ds_gcm, method="conservative_normed")
    Era5_new = ds_era5_to_gcm.tas.values.astype(np.float64)
    
    # 3. Numerical Certification
    print(f"\n[Certification: 8x13 GCM Grid]")
    mask = ~np.isnan(Era5_new) & ~np.isnan(Era5_base)
    diff = np.abs(Era5_new[mask] - Era5_base[mask])
    max_diff = np.nanmax(diff)
    print(f"Predictor MaxDiff: {max_diff:.4e} K")
