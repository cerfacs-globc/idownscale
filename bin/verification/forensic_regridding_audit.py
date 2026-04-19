'''
Forensic Regridding Audit: Pre-Interpolation Logic
Tests the archival "Bilinear-before-Conservative" sequence.
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
from iriscc.datautils import Data
from iriscc.settings import DATASET_EXP5_30Y_DIR, CONFIG

def forensic_audit(date_str):
    date = pd.Timestamp(date_str)
    exp = 'exp5'
    
    # 1. Load Baseline
    baseline_path = DATASET_EXP5_30Y_DIR / f"sample_{date.strftime('%Y%m%d')}.npz"
    baseline = np.load(baseline_path)
    X_base = baseline['x'][1] # Temperature channel
    
    # 2. Setup
    get_data = Data(domain=CONFIG[exp]['domain'])
    from iriscc.datautils import interpolation_target_grid, reformat_as_target
    
    print(f"--- Archival Pre-Interpolation Forensic ({date_str}) ---")
    
    # Load assets
    ds_era5 = get_data.get_era5_dataset('tas', date)
    ds_gcm = get_data.get_gcm_dataset('tas', date, CONFIG[exp]['ssp'])
    
    # Archival Logic Recovery: Pre-interpolation if finer
    print("\nExecuting Archival Pre-Interpolation (Bilinear -> GCM Size)...")
    for coord in ['lat', 'lon']:
        if len(ds_era5[coord]) > len(ds_gcm[coord]):
            # Fixed target size matching GCM
            new_coord = np.linspace(ds_era5[coord].values.min(), ds_era5[coord].values.max(), len(ds_gcm[coord]))
            ds_era5 = ds_era5.interp({coord: new_coord})
    
    # Stage 1: ERA5 (Interpolated) -> GCM (Conservative)
    ds_era5_to_gcm = interpolation_target_grid(ds_era5, ds_target=ds_gcm, method="conservative_normed")
    
    # Stage 2: GCM -> E-OBS
    ds_new = reformat_as_target(ds_era5_to_gcm, 
                                target_file=CONFIG[exp]['target_file'], 
                                method='conservative_normed', 
                                domain=CONFIG[exp]['domain'], 
                                crop_target=True, mask=True)
    X_new = ds_new['tas'].values
    
    # 3. Audit comparison
    mask = ~np.isnan(X_new) & ~np.isnan(X_base)
    diff = np.abs(X_new[mask] - X_base[mask]).max()
    
    print(f"\nAudit Result (Pre-Interp): Max Diff = {diff:.2e} K")
    if diff < 1e-10:
        print("\n[SUCCESS] Archival Pre-Interpolation methodology matches bit-identically!")

if __name__ == "__main__":
    forensic_audit("1980-01-01")
