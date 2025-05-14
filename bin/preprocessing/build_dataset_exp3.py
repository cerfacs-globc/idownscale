''' Experience 2 : ERA5 and topography as input and SAFRAN as target'''

import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import pandas as pd
import glob
from datetime import datetime

from iriscc.plotutils import plot_test
from iriscc.datautils import (standardize_dims_and_coords, 
                              standardize_longitudes, 
                              interpolation_target_grid, 
                              reformat_as_target,
                              remove_countries,
                              crop_domain_from_ds)
from iriscc.settings import (DATES,
                             DATES_TEST,
                             CONFIG,
                             ERA5_DIR,
                             DATASET_EXP3_30Y_DIR,
                             TARGET,
                             SAFRAN_REFORMAT_DIR,
                             OROG_FILE,
                             CMIP6_RAW_DIR,
                             TARGET_SAFRAN_FILE,
                             INPUTS)

def get_era5_dataset(date, domain):
    file = glob.glob(str(ERA5_DIR/f'tas*_{date.year}_*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_dims_and_coords(ds)
    ds = standardize_longitudes(ds)
    ds = ds.reindex(lat=ds.lat[::-1])
    ds = crop_domain_from_ds(ds, domain)
    ds = ds.sel(time=ds.time.dt.date == date.date())
    ds = ds.isel(time=0)
    return ds

def get_cmip6_dataset(domain):
    file = glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_longitudes(ds)
    ds = crop_domain_from_ds(ds, domain)
    ds = ds.isel(time=0)
    return ds

def input_data(date, domain):
    ''' Returns inputs data as an array of shape (C, H, W) '''
    x = []

    # Commune variables
    ds = xr.open_dataset(OROG_FILE) # Already interpolated to target grids
    x.append(ds['Altitude'].values)

    for var in INPUTS:
        ds_era5 = get_era5_dataset(date, domain)
        ds_cmip6 = get_cmip6_dataset(domain)
        ds_era5_to_cmip6 = interpolation_target_grid(ds_era5, 
                                                     ds_target=ds_cmip6, 
                                                     method="conservative_normed")
        ds = reformat_as_target(ds_era5_to_cmip6, 
                                target_file=TARGET_SAFRAN_FILE, 
                                method='conservative_normed',
                                domain = CONFIG['safran']['domain']['france'],
                                crop_target=False)
        x.append(ds[var].values)
    x = np.concatenate([x[0][np.newaxis, :, :]] + [x[1][np.newaxis, :, :]] * 1, axis=0)

    return x



def target_data(date):
    ''' Returns target data as an array of shape (H, W) '''
    ds = xr.open_dataset(glob.glob(str(SAFRAN_REFORMAT_DIR/f"tas*{date.year}_reformat.nc"))[0])
    ds.sel(time=ds.time.dt.date == date.date())
    ds = ds.isel(time=0)
    y = ds[TARGET].values
    y = remove_countries(y)
    y = np.expand_dims(y, axis=0)
    return y


if __name__=='__main__':
    domain = CONFIG['safran']['domain']['france']

    for date in DATES:
        print(date)

        x = input_data(date, domain)
        y = target_data(date)

        sample = {'x' : x,
                  'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_EXP3_30Y_DIR/f'sample_{date_str}.npz', **sample)
         



      