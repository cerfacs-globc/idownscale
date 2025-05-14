''' Experience 2 : ERA5 and topography as input and SAFRAN as target'''

import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import pandas as pd
import glob
from datetime import datetime

from iriscc.datautils import (standardize_dims_and_coords, 
                              standardize_longitudes, 
                              interpolation_target_grid, 
                              reformat_as_target,
                              remove_countries,
                              apply_landseamask)
from iriscc.settings import (
                             DATES_TEST,
                             ERA5_DIR,
                             DATASET_EXP3_BASELINE_DIR,
                             TARGET,
                             SAFRAN_REFORMAT_DIR,
                             CMIP6_RAW_DIR,
                             TARGET_SAFRAN_FILE,
                             CONFIG,
                             INPUTS)

def get_era5_dataset(date):
    file = glob.glob(str(ERA5_DIR/f'tas*_{date.year}_*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_dims_and_coords(ds)
    ds = standardize_longitudes(ds)
    ds = ds.reindex(lat=ds.lat[::-1])
    ds = ds.sel(lon=slice(-6,12), lat=slice(40.,52.))
    ds = ds.sel(time=ds.time.dt.date == date.date())
    ds = ds.isel(time=0)
    return ds

def get_cmip6_dataset():
    file = glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_longitudes(ds)
    ds = apply_landseamask(ds, 'cmip6')
    ds = ds.sel(lon=slice(-6,12), lat=slice(40.,52.))
    ds = ds.isel(time=0)
    return ds

def prediction_data(date):
    ''' Returns inputs data as an array of shape (C, H, W) '''

    ds_era5 = get_era5_dataset(date)
    ds_cmip6 = get_cmip6_dataset()
    ds_era5_to_cmip6 = interpolation_target_grid(ds_era5, 
                                                 ds_target=ds_cmip6, 
                                                 method="conservative_normed")
    ds = reformat_as_target(ds_era5_to_cmip6, 
                            target_file=TARGET_SAFRAN_FILE, 
                            method='bilinear',
                            domain=CONFIG['safran']['domain'],
                            crop_target=False)

    y_hat = ds[TARGET].values
    y_hat = remove_countries(y_hat)
    return y_hat



def target_data(date):
    ''' Returns target data as an array of shape (H, W) '''
    ds = xr.open_dataset(glob.glob(str(SAFRAN_REFORMAT_DIR/f"tas*{date.year}_reformat.nc"))[0])
    ds.sel(time=ds.time.dt.date == date.date())
    ds = ds.isel(time=0)
    y = ds[TARGET].values
    y = remove_countries(y)
    return y


if __name__=='__main__':

    for date in DATES_TEST:
        print(date)

        y_hat = prediction_data(date)
        y = target_data(date)

        sample = {'y_hat' : y_hat,
                  'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_EXP3_BASELINE_DIR/f'sample_{date_str}.npz', **sample)
         



      