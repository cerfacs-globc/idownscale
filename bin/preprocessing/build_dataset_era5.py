import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import pandas as pd
import glob
from datetime import datetime

from iriscc.plotutils import plot_test
from iriscc.datautils import standardize_dims_and_coords, standardize_longitudes, interpolation_target_grid, reformat_as_target
from iriscc.settings import (DATES,
                             ERA5_DIR,
                             DATASET_EXP2_6MB_DIR,
                             TARGET,
                             SAFRAN_DIR,
                             OROG_FILE,
                             CMIP6_RAW_DIR,
                             TARGET_GRID_FILE,
                             INPUTS)

def get_era5_dataset(date):
    file = glob.glob(str(ERA5_DIR/f'*{date.year}*'))[0]
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
    ds = ds.sel(lon=slice(-6,12), lat=slice(40.,52.))
    ds = ds.isel(time=0)
    return ds

def input_data(date):
    ''' Returns inputs data as an array of shape (C, H, W) '''
    x = []

    # Commune variables
    ds = xr.open_dataset(OROG_FILE) # Already interpolated to target grids
    x.append(ds['z'].values)
    plot_test(ds['z'].values, 'z', '/scratch/globc/garcia/graph/test2.png')

    for var in INPUTS:
        ds_era5 = get_era5_dataset(date)
        ds_cmip6 = get_cmip6_dataset()
        ds_era5_to_cmip6 = interpolation_target_grid(ds_era5, ds_target=ds_cmip6)
        ds = reformat_as_target(ds_era5_to_cmip6, target_file=TARGET_GRID_FILE)
        x.append(ds[var].values)
    x = np.concatenate([x[0][np.newaxis, :, :]] + [x[1][np.newaxis, :, :]] * 6, axis=0)

    return x



def target_data(date):
    ''' Returns target data as an array of shape (H, W) '''

    threshold_date = datetime(date.year, 8, 1)
    year = date.year
    if date < threshold_date : 
        year = year-1
    ds = xr.open_dataset(glob.glob(str(SAFRAN_DIR/f"SAFRAN_{year}080107_{year+1}080106_reformat.nc"))[0])

    if date == datetime(date.year, 8, 1):
        ds_before = xr.open_dataset(glob.glob(str(SAFRAN_DIR/f"SAFRAN_{year-1}080107_{year}080106_reformat.nc"))[0])
        ds_before = ds_before.isel(time=slice (-7, None))
        ds = ds.isel(time = slice(None, 17))
        ds = ds.merge(ds_before)
    else : 
        ds = ds.sel(time=pd.date_range(start=date.strftime("%Y-%m-%d"), periods = 24, freq='h').to_numpy())

    y = ds[TARGET].values.mean(axis=0)
    y = np.expand_dims(y, axis=0)
    return y


if __name__=='__main__':

    for date in DATES:
        print(date)

        x = input_data(date)
        y = target_data(date)

        sample = {'x' : x,
                  'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_EXP2_6MB_DIR/f'sample_{date_str}.npz', **sample)
         



      