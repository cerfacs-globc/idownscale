import xarray as xr
import numpy as np
import os

def check_file(label, path):
    if not os.path.exists(path):
        print(f"{label} file NOT FOUND: {path}")
        return None
    ds = xr.open_dataset(path)
    # Select year 2010 if it spans multiple years
    if 'time' in ds.dims:
        try:
            ds = ds.sel(time='2010')
        except:
            pass
    mean_val = ds.tas.mean().values
    print(f"{label} Mean Temp (2010): {mean_val:.4f} K")
    return mean_val

our_file = "/scratch/globc/page/idownscale_garcia_clean/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20100101_20101231_exp5_unet_gcm_bc.nc"
ref_file = "/scratch/globc/garcia/idownscale/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101_20141231_exp5_unet_all_gcm_bc.nc"

print("--- SCIENTIFIC PARITY COMPARISON (2010) ---")
check_file("Our Version_0", our_file)
check_file("Garcia Reference", ref_file)
