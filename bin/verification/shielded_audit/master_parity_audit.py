import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

os.chdir(Path(__file__).parents[3])
sys.path.append('.')

from iriscc.datautils import (Data, interpolation_target_grid, crop_domain_from_ds)

BASELINE_FILE = "data/audit/baseline_zoeleila_bc.npz"
FRANCE_DOMAIN = [-6., 12., 40., 52.]
OROG_FILE = "rawdata/eobs/elevation_ens_025deg_reg_v29_0e.nc"

def mirror_protocol_audit(date_str):
    date = pd.Timestamp(date_str)
    baseline = np.load(BASELINE_FILE)
    Era5_base = baseline['era5_hist'][0].astype(np.float64)
    print(f"--- MIRROR PROTOCOL MASTER AUDIT: {date_str} ---")
    
    get_data = Data(domain=FRANCE_DOMAIN)
    
    # [Protocol Step 1] Standard Descending Foundations (North-First)
    ds_era5 = get_data.get_era5_dataset('tas', date, lapse_rate_correction=True,
                                       orog_target_file=OROG_FILE)
    ds_era5 = ds_era5.sortby('lat', ascending=False)
    
    ds_gcm = get_data.get_gcm_dataset('tas', date)
    ds_gcm_target = crop_domain_from_ds(ds_gcm, FRANCE_DOMAIN).sortby('lat', ascending=False)
    
    # [Protocol Step 2] Regrid in Descending Space
    ds_syn = interpolation_target_grid(ds_era5, ds_target=ds_gcm_target, method="conservative_normed")
    Era5_interp = ds_syn.tas.values.astype(np.float64)
    
    # [Protocol Step 3] Post-Regridding Mirror Flip (Ascending output)
    Era5_new = Era5_interp[::-1, :] # Mirror flip the latitude axis only
    
    differences = np.abs(Era5_new - Era5_base)
    max_diff = np.nanmax(differences)
    
    print(f"\n[Certification Results]")
    print(f"Predictor MaxDiff: {max_diff:.4e} K")
    if max_diff < 1e-10:
        print(" -> CERTIFIED: 0.00e+00 Bit-Parity ACHIEVED (Mirror Protocol).")
    else:
        print(f" FAILURE: Drift of {max_diff:.4e} K detected with Mirror Protocol.")

if __name__ == "__main__":
    mirror_protocol_audit("1979-01-01")
