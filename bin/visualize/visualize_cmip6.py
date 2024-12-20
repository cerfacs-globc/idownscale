import sys
sys.path.append('.')

import xarray as xr
from datetime import datetime
import numpy as np

from iriscc.plotutils import plot_map
from iriscc.settings import CMIP6_RAW_DIR
from iriscc.datautils import standardize_longitudes


ds = xr.open_dataset(CMIP6_RAW_DIR/'tas_day_CNRM-CM6-1_historical_r10i1p1f2_gr_18500101-20141231.nc')
ds = standardize_longitudes(ds)
ds = ds.sel(lon=slice(-5.625,11.25))
ds = ds.sel(lat=slice(39.,52.))
ds = ds.isel(time=0)

time = ds['time'].values

date = time.astype('datetime64[s]').astype(datetime)
date = date.strftime('%Y%m%d')
lon = ds['lon'].values
lat = ds['lat'].values
lon_grid, lat_grid = np.meshgrid(lon, lat)
plot_map(lon_grid,
            lat_grid,
            ds['tas'].values - 273.15,
            f'Near Surface Temperature {date} (CNRM-CM6-1 r10i1p1f2)',
            f'/scratch/globc/garcia/graph/CNRM-CM6_r10i1p1f2_tas_{date}.png')
