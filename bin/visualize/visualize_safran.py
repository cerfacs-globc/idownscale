import sys
sys.path.append('.')

import xarray as xr
from datetime import datetime
import numpy as np

from iriscc.plotutils import plot_map


file = xr.open_dataset('/scratch/globc/garcia/rawdata/safran/SAFRAN_1986080107_1987080106_reformat.nc')
tas = file['tas'].values[1]
time = file['time'].values[1]
date = time.astype('datetime64[s]').astype(datetime)
date = date.strftime('%Y%m%d-%H')
lon = file['lon'].values
lat = file['lat'].values
tas_mask = np.ma.masked_where(tas == -9999, tas)
plot_map(lon,
            lat,
            tas_mask - 273.15,
            f'Near Surface Temperature {date} (SAFRAN)',
            f'/scratch/globc/garcia/graph/SAFRAN_tas_{date}.png')

