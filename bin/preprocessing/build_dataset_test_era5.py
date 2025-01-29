import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import glob

from iriscc.datautils import standardize_dims_and_coords, standardize_longitudes
from iriscc.settings import (DATES_TEST,
                             ERA5_DIR,
                             DATASET_TEST_ERA5_DIR)

def get_era5_dataset(date):
    file = glob.glob(str(ERA5_DIR/f'*{date.year}*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_dims_and_coords(ds)
    ds = standardize_longitudes(ds)
    ds = ds.reindex(lat=ds.lat[::-1])
    ds = ds.sel(lon=slice(-5,12), lat=slice(41.,51.))
    ds = ds.sel(time=ds.time.dt.date == date.date())
    ds = ds.isel(time=0)
    return ds


if __name__=='__main__':

    for date in DATES_TEST:
        print(date)
        ds_era5 = get_era5_dataset(date)
        y_era5 = ds_era5.tas.values
        sample = {'y_era5' : y_era5}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_TEST_ERA5_DIR/f'sample_{date_str}.npz', **sample)



      