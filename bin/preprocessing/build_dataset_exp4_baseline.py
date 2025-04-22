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
                              crop_domain_from_ds,
                              apply_landseamask)
from iriscc.settings import (DATES_TEST,
                             ERA5_DIR,
                             DATASET_EXP4_BASELINE_DIR,
                             TARGET,
                             EOBS_RAW_DIR,
                             CMIP6_RAW_DIR,
                             TARGET_EOBS_FILE,
                             CONFIG,
                             INPUTS)

def get_era5_dataset(date, domain):
    file = glob.glob(str(ERA5_DIR/f'tas*_{date.year}_*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_dims_and_coords(ds)
    ds = standardize_longitudes(ds)
    ds = ds.reindex(lat=ds.lat[::-1])
    ds = ds.sel(time=ds.time.dt.date == date.date()).isel(time=0)
    ds = crop_domain_from_ds(ds, domain)
    return ds

def get_cmip6_dataset(domain):
    file = glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_longitudes(ds)
    ds = ds.isel(time=0)
    ds = crop_domain_from_ds(ds, domain)
    return ds

def prediction_data(date, domain):
    ''' Returns inputs data as an array of shape (C, H, W) '''
    ds_era5 = get_era5_dataset(date, domain)
    ds_cmip6 = get_cmip6_dataset(domain)
    ds_era5_to_cmip6 = interpolation_target_grid(ds_era5, 
                                                 ds_target=ds_cmip6, 
                                                 method="conservative_normed")

    ds = reformat_as_target(ds_era5_to_cmip6, 
                            target_file=TARGET_EOBS_FILE, 
                            method='bilinear',
                            domain=domain,
                            crop_target=True)

    y_hat = ds[TARGET].values
    return y_hat



def target_data(date, domain):
    ''' Returns target data as an array of shape (H, W) '''
    file = glob.glob(str(EOBS_RAW_DIR/f'tas*'))[0]
    ds = xr.open_dataset(file)
    ds = ds.sel(time=ds.time.dt.date == date.date()).isel(time=0)
    ds = standardize_dims_and_coords(ds)
    ds = apply_landseamask(ds, 'eobs')
    ds = crop_domain_from_ds(ds, domain)
    lon = ds['lon'].values
    lat = ds['lat'].values
    y = ds[TARGET].values
    if np.nanmean(y) < 100: # if celsus
        y = y + 273.15
    return y, lon, lat

if __name__=='__main__':
    domain = CONFIG['eobs']['domain']['europe']
    for i, date in enumerate(DATES_TEST):
        print(date)

        y_hat = prediction_data(date, domain)
        
        if i == 0:
            y, lon, lat = target_data(date, domain)
            coordinates = {'lon': lon,
                            'lat': lat}
            np.savez(DATASET_EXP4_BASELINE_DIR/f'coordinates.npz', **coordinates)
        else:
            y, _, _ = target_data(date, domain)

        sample = {'y_hat' : y_hat,
                  'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_EXP4_BASELINE_DIR/f'sample_{date_str}.npz', **sample)
         



      