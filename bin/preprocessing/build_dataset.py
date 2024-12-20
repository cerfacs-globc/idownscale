import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import glob
import numpy.ma as ma
from datetime import datetime
import pandas as pd
import json

from iriscc.plotutils import plot_image
from iriscc.datautils import reformat_as_target
from iriscc.settings import (SAFRAN_DIR, 
                             CMIP6_RAW_DIR,
                             DATES,
                             INPUTS,
                             GCM,
                             OROG_FILE,
                             IMERG_MASK,
                             TARGET_GRID_FILE,
                             TARGET,
                             CHANELS,
                             DATASET_EXP1_CONTINENTS_DIR,
                             DATASET_EXP1_DIR)


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
                ds[var] = ds[var].transpose()
                ds = reformat_as_target(ds)
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
    y = np.expand_dims(y, axis=0)
    return y



def update_statistics(sum, square_sum, n_total, x):
    ''' Compute and update samples statistics '''
    x = x[~np.isnan(x)]
    sum += np.sum(x)
    square_sum += np.sum(x**2)
    n_total += x.size
    return sum, square_sum, n_total





if __name__=='__main__':

    dates = DATES

    ch = len(CHANELS)
    sum = np.zeros(ch)
    square_sum = np.zeros(ch)
    n_total = np.zeros(ch)

    for date in dates:
        print(date)

        x = input_data(date.date())
        y = target_data(date)

        sample = {'x' : x,
                    'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_EXP1_DIR/f'sample_{date_str}.npz', **sample)

        for i in range(ch):
            sum[i], square_sum[i], n_total[i] = update_statistics(sum[i], 
                                                                    square_sum[i], 
                                                                    n_total[i],
                                                                    x[i])

    mean = sum / n_total
    std = np.sqrt((square_sum / n_total) - (mean**2))
    print(mean, std)

    stats = {}
    for i, chanel in enumerate(CHANELS):
        stats[chanel] = {'mean': mean[i],
                         'std': std[i]}

    with open(DATASET_EXP1_DIR/'statistics.json', "w") as f: 
	    json.dump(stats, f)

      