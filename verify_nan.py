import xarray as xr
import numpy as np

f_era5 = '/scratch/globc/page/idownscale_rerun/rawdata/era5/orography_ERA5.nc'
f_eobs = '/scratch/globc/page/idownscale_rerun/rawdata/eobs/elevation_ens_025deg_reg_v29_0e.nc'

ds_z = xr.open_dataset(f_era5).isel(time=0)
ds_target = xr.open_dataset(f_eobs)

h1 = ds_z['z'] / 9.80665
h2 = ds_target['elevation']

print(f"ERA5 NaNs: {h1.isnull().sum().values}")
print(f"EOBS NaNs: {h2.isnull().sum().values}")

# Rough alignment
h2_aligned = h2.reindex_like(h1, method='nearest')
delta = h1 - h2_aligned
print(f"Delta NaNs (Raw): {delta.isnull().sum().values}")

# With fillna
delta_filled = h1.fillna(0) - h2_aligned.fillna(0)
print(f"Delta NaNs (Filled): {delta_filled.isnull().sum().values}")
