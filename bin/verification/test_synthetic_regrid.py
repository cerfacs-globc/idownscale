'''
Synthetic Regridding Diagnostic 2.0
Tests the sensitivity of the two-step regridding pipeline
using controlled synthetic data.

date : 19/04/2026
author : Antigravity (AI Assistant)
'''
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2].resolve()
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import xarray as xr
import pandas as pd
from iriscc.datautils import Data, interpolation_target_grid, reformat_as_target
from iriscc.settings import CONFIG

def test_synthetic():
    exp = 'exp5'
    domain = CONFIG[exp]['domain']
    get_data = Data(domain=domain)
    
    # 1. Create a Synthetic ERA5 (Source) with a strong gradient
    ds_era5 = get_data.get_era5_dataset('tas', pd.Timestamp("1980-01-01"))
    ds_syn = ds_era5.copy(deep=True)
    # Correct broadcasting: (Lat, Lon)
    lat_grad = np.broadcast_to(ds_syn.lat.values[:, None], ds_syn['tas'].shape)
    ds_syn['tas'].values = lat_grad + 273.15
    
    print("--- Synthetic Regridding Sensitivity (Broadcast Fixed) ---")
    
    # 2. Intermediate GCM
    ds_gcm = get_data.get_gcm_dataset('tas', pd.Timestamp("1980-01-01"), CONFIG[exp]['ssp'])
    
    print("\n[Audit] Coordinate Precision:")
    print(f"ERA5 Lat Step: {np.diff(ds_era5.lat.values[:2])[0]:.8f}")
    print(f"GCM Lat Step:  {np.diff(ds_gcm.lat.values[:2])[0]:.8f}")

    # 3. Reference Regridding (Base case)
    print("\nRunning Two-Step Regridding (Ref)...")
    ds_ref_gcm = interpolation_target_grid(ds_syn, ds_target=ds_gcm, method="conservative_normed")
    ds_ref_final = reformat_as_target(ds_ref_gcm, 
                                      target_file=CONFIG[exp]['target_file'], 
                                      method='conservative_normed', 
                                      domain=domain, 
                                      crop_target=True, mask=True)
    X_ref = ds_ref_final['tas'].values

    # 4. Hypothesis: 0.1 degree GCM offset (Center vs Corner)
    ds_gcm_s = ds_gcm.copy(deep=True)
    ds_gcm_s = ds_gcm_s.assign_coords(lat=ds_gcm_s.lat + 0.1)
    
    ds_s_gcm = interpolation_target_grid(ds_syn, ds_target=ds_gcm_s, method="conservative_normed")
    ds_s_final = reformat_as_target(ds_s_gcm, 
                                    target_file=CONFIG[exp]['target_file'], 
                                    method='conservative_normed', 
                                    domain=domain, 
                                    crop_target=True, mask=True)
    X_s = ds_s_final['tas'].values
    
    diff = np.nanmax(np.abs(X_s - X_ref))
    print(f"Max Diff for 0.1 degree GCM Lat Shift: {diff:.2f} K")
    
    # 5. Self-Regridding Test (Identity Check)
    print("\nRunning Identity Check (GCM -> GCM)...")
    ds_ident = interpolation_target_grid(ds_gcm, ds_target=ds_gcm, method="conservative_normed")
    diff_ident = np.nanmax(np.abs(ds_ident['tas'].values - ds_gcm['tas'].values))
    print(f"Max Diff (GCM -> GCM): {diff_ident:.2e}")

if __name__ == "__main__":
    test_synthetic()
