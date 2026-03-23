import xarray as xr
import sys

def check(name, path, var):
    try:
        ds = xr.open_dataset(path)
        if 'time' in ds.dims:
            ds = ds.isel(time=0)
        shape = ds[var].shape
        print(f"{name} ({var}): {shape}")
    except Exception as e: # noqa: BLE001
        print(f"Error checking {name}: {e}")

check("Orog v31", "/scratch/globc/page/idownscale_active/rawdata/eobs/elevation_ens_025deg_reg_v31_0e_france.nc", "elevation")
check("Target v29", "/scratch/globc/page/idownscale_active/rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc", "tas")
check("Target Safran", "/scratch/globc/page/idownscale_active/rawdata/safran/safran_reformat_day/tas_day_SAFRAN_1959_reformat.nc", "tas")
