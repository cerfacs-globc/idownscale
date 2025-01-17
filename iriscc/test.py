import sys
sys.path.append('.')

import glob
import numpy as np
import xarray as xr
from iriscc.settings import DATASET_EXP1_DIR
from iriscc.plotutils import plot_test
from iriscc.datautils import standardize_longitudes, standardize_dims_and_coords

file = '/scratch/globc/garcia/rawdata/era5/tas_1m_194001-202312_ERA5.nc'

ds = xr.open_dataset(file, engine='netcdf4')
ds = standardize_dims_and_coords(ds)
ds = standardize_longitudes(ds)
ds = ds.reindex(lat=ds.lat[::-1])
print(ds)
ds = ds.sel(lon=slice(-6,12), lat=slice(40.,52.))

ds= ds.isel(time=-1)

tas = ds['tas'].values
print(tas.shape)
plot_test(tas, 'test era5', '/scratch/globc/garcia/graph/test.png')