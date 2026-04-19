'''
ETOPO Lapse Rate Forensic Verification Script
Tests if pre-processing ERA5 with an altitude correction based on the 
authentic archival ETOPO grid resolves the 12.3 K bias.

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
from iriscc.datautils import (Data, interpolation_target_grid, 
                             reformat_as_target, standardize_dims_and_coords)
from iriscc.settings import DATASET_EXP5_30Y_DIR, CONFIG

def verify_etopo_lapse(date_str):
    date = pd.Timestamp(date_str)
    exp = 'exp5'
    
    # 1. Load Archival Benchmark
    baseline_path = DATASET_EXP5_30Y_DIR / f"sample_{date.strftime('%Y%m%d')}.npz"
    baseline = np.load(baseline_path)
    X_base = baseline['x'][1] # Temperature channel
    
    # 2. Setup Assets
    get_data = Data(domain=CONFIG[exp]['domain'])
    
    # A. Authentic ETOPO Topography (Target Correction)
    etopo_file = "/scratch/globc/page/idownscale_rerun/rawdata/topography/ETOPO_2022_v1_30s_N90W180_bed_regrid.nc"
    ds_etopo = xr.open_dataset(etopo_file)
    H_etopo = ds_etopo['z'] # Units: meters
    
    # B. Archival ERA5 Geopotential (Source Correction)
    z_file = "/scratch/globc/page/idownscale_rerun/rawdata/era5/orography_ERA5.nc"
    ds_z = xr.open_dataset(z_file).isel(time=0)
    ds_z = standardize_dims_and_coords(ds_z)
    H_era5 = ds_z['z'] / 9.80665
    
    # C. Raw ERA5 Temperature
    ds_era5 = get_data.get_era5_dataset('tas', date)
    
    print(f"--- ETOPO Lapse Rate Forensic ({date_str}) ---")
    
    # 3. APPLY PRE-PROCESSING (The "Archival ETOPO Process")
    # T_adj = T_raw - (H_etopo - H_era5) * 0.0065
    print("Applying High-Res ETOPO Altitude Correction (-0.0065 K/m)...")
    import xesmf as xe
    
    # Create Regridder for ERA5 -> ETOPO (Curvilinear 134x143)
    regridder = xe.Regridder(H_era5, ds_etopo, 'bilinear')
    H_era5_target = regridder(H_era5)
    
    # Delta (HighRes ETOPO - Source ERA5) on the Target Grid
    delta_h = H_etopo - H_era5_target
    correction = delta_h * (-0.0065)
    
    # To apply correction to ERA5, we average it DOWN to the ERA5 grid
    regridder_back = xe.Regridder(ds_etopo, ds_era5, 'bilinear')
    correction_coarse = regridder_back(correction)
    
    ds_era5['tas'].values = ds_era5['tas'].values + correction_coarse.values
    
    # 4. RUN PRODUCTION REGRIDDING (2-Step)
    ds_gcm = get_data.get_gcm_dataset('tas', date, CONFIG[exp]['ssp'])
    ds_era5_to_gcm = interpolation_target_grid(ds_era5, 
                                            ds_target=ds_gcm, 
                                            method="conservative_normed")
    ds_final = reformat_as_target(ds_era5_to_gcm, 
                                    target_file=CONFIG[exp]['target_file'], 
                                    method='conservative_normed', 
                                    domain=CONFIG[exp]['domain'], 
                                    crop_target=True, mask=True)
    X_new = ds_final['tas'].values
    
    # 5. AUDIT
    mask = ~np.isnan(X_new) & ~np.isnan(X_base)
    diff = np.abs(X_new[mask] - X_base[mask]).max()
    mean = (X_new[mask] - X_base[mask]).mean()
    
    print(f"\nAudit Result (ETOPO-Corrected):")
    print(f"Max Difference: {diff:.2e} K")
    print(f"Mean Bias:      {mean:.2e} K")
    
    if diff < 1.0:
        print("\n[VERDICT] ETOPO altitude correction confirmed as the archival methodology.")

if __name__ == "__main__":
    verify_etopo_lapse("1980-01-01")
