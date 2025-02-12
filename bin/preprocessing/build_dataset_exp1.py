''' Experience 1 : CMIP6 and topography as input and SAFRAN as target'''

import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import glob
import numpy.ma as ma
from datetime import datetime
import pandas as pd
import json

from iriscc.plotutils import plot_test
from iriscc.datautils import reformat_as_target, standardize_longitudes
from iriscc.settings import (SAFRAN_DIR, 
                             CMIP6_RAW_DIR,
                             DATES,
                             INPUTS,
                             GCM,
                             OROG_FILE,
                             IMERG_MASK,
                             TARGET_GRID_FILE,
                             TARGET,
                             DATASET_EXP1_30Y_DIR,
                             DATASET_EXP1_DIR, 
                             DATASET_EXP1_6MB_DIR,
                             DATASET_EXP1_6MB_30Y_DIR)


def mask_coverage_func(var_array, mask, model):
    ''' Create a mask on an input array to remove sea and/or continental values '''

    if mask == 'france':
        ds = xr.open_dataset(TARGET_GRID_FILE)
        ds = ds.isel(time=0)
        condition = np.isnan(ds['tas'].values)
    elif mask == 'continents':
        ds = xr.open_dataset(IMERG_MASK)
        ds['landseamask'].values = 100 - ds['landseamask'].values
        condition = ds['landseamask'].values < 25
    elif mask == 'coarse':
        ds = xr.open_dataset(glob.glob(str(CMIP6_RAW_DIR/f'{model}/sftlf*'))[0])
        condition = ds['sftlf'].values < 2

    var_array[condition] = np.nan
    mask_var_array = var_array

    return mask_var_array



def input_data(date):
    ''' Returns inputs data as an array of shape (C, H, W) '''

    x = []

    # Commune variables
    ds = xr.open_dataset(OROG_FILE)
    #z = mask_coverage_func(ds['z'].values, 'france', None)
    x.append(ds['z'].values)

    # Model variables
    for var in INPUTS:
        for model in GCM:
            ensemble = np.sort(glob.glob(str(CMIP6_RAW_DIR/f'{model}/{var}*')))
            for member in ensemble:
                ds = xr.open_dataset(member)
                ds = ds.sel(time=ds.time.dt.date == date)
                ds = ds.isel(time=0)
                #if mask == 'coarse':
                    #ds[var].values = mask_coverage_func(ds[var].values, mask, model)
                ds = standardize_longitudes(ds)
                ds = ds.sel(lon=slice(-6,12), lat=slice(40.,52.))
                plot_test(ds['tas'].values, 'tas (K) 20040101 CNRM-CM6-1', '/scratch/globc/garcia/graph/test.png', vmin = 262, vmax = 286)
                ds[var] = ds[var].transpose()
                print('coucou')
                ds = reformat_as_target(ds, target_file=TARGET_GRID_FILE, method="conservative_normed")
                plot_test(ds['tas'].values, 'tas (K) 20040101 CNRM-CM6-1 interpolated', '/scratch/globc/garcia/graph/test4.png', vmin = 262, vmax = 286)
                #if mask == 'continents' or mask == 'france':
                    #ds[var].values = mask_coverage_func(ds[var].values, mask, None)
                x.append(ds[var].values)
    x = np.stack(x, axis = 0)
    return x


def target_data(date):
    ''' Returns target data as an array of shape (C, H, W) '''

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
    plot_test(y, 'tas (K) 20040101 SAFRAN', '/scratch/globc/garcia/graph/test3.png',vmin = 262, vmax = 286)
    y = np.expand_dims(y, axis=0)
    return y



if __name__=='__main__':

    dates = DATES[0:1]
    for date in dates:
        print(date)

        x = input_data(date.date())
        y = target_data(date)

        sample = {'x' : x,
                    'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        #np.savez(DATASET_EXP1_6MB_30Y_DIR/f'sample_{date_str}.npz', **sample)



      