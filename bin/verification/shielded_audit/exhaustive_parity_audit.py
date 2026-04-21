import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).parents[3].resolve()
sys.path.append(str(PROJECT_ROOT))

from iriscc.datautils import (Data, interpolation_target_grid, crop_domain_from_ds)

BASELINE_FILE = PROJECT_ROOT / "data/audit/baseline_zoeleila_bc.npz"
FRANCE_DOMAIN = [-6., 12., 40., 52.]
OROG_FILE = PROJECT_ROOT / "rawdata/eobs/elevation_ens_025deg_reg_v29_0e.nc"

def exhaustive_audit(date_str):
    date = pd.Timestamp(date_str)
    baseline = np.load(BASELINE_FILE)
    Era5_base = baseline['era5_hist'][0].astype(np.float64)
    print(f"--- EXHAUSTIVE PARITY SEARCH: {date_str} ---")
    
    # 1. Generating Native foundations (France Clipped)
    get_data = Data(domain=FRANCE_DOMAIN)
    ds_era5 = get_data.get_era5_dataset('tas', date, lapse_rate_correction=True,
                                       orog_target_file=str(OROG_FILE))
    ds_gcm = get_data.get_gcm_dataset('tas', date)
    ds_gcm_france = crop_domain_from_ds(ds_gcm, FRANCE_DOMAIN)
    
    # Use exact regridding to GCM anchors
    ds_syn = interpolation_target_grid(ds_era5, ds_target=ds_gcm_france,  method="conservative_normed")
    Era5_new = ds_syn.tas.values.astype(np.float64)
    
    # 2. Testing 8 Orientations (Flips/Transposes)
    orientations = {
        "Native": Era5_new,
        "Flip-Lat": Era5_new[::-1, :],
        "Flip-Lon": Era5_new[:, ::-1],
        "Flip-Both": Era5_new[::-1, ::-1],
        "Celsius-Native": Era5_new - 273.15,
        "Celsius-Flip-Lat": Era5_new[::-1, :] - 273.15,
        "Celsius-Flip-Lon": Era5_new[:, ::-1] - 273.15,
        "Celsius-Flip-Both": Era5_new[::-1, ::-1] - 273.15
    }
    
    mask = ~np.isnan(Era5_base)
    found_parity = False
    for name, data in orientations.items():
        if data.shape != Era5_base.shape: continue
        diff = np.abs(data[mask] - Era5_base[mask])
        max_diff = np.nanmax(diff)
        print(f"  [{name:18s}] MaxDiff: {max_diff:.4e} K")
        if max_diff < 1e-10:
            print(f"\n  -> CERTIFIED PARITY ACHIEVED: {name}")
            found_parity = True
            break
            
    if not found_parity:
        print("\n  -> FAILURE: Parity not found in Standard Orientation/Unit matrix.")
        print(f"  Archive Row 0 Mean: {Era5_base[0, :].mean():.2f}")
        print(f"  Archive Row -1 Mean: {Era5_base[-1, :].mean():.2f}")

if __name__ == "__main__":
    exhaustive_audit("1979-01-01")
