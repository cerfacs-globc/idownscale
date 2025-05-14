''' Data preprocessing for bias correction'''

import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import pandas as pd
import glob

from iriscc.datautils import (standardize_dims_and_coords, 
                              standardize_longitudes, 
                              interpolation_target_grid, 
                              crop_domain_from_ds)
from iriscc.settings import (DATES_TEST,
                             DATES_BC_TRAIN_HIST,
                             DATES_BC_TEST_HIST,
                             DATES_BC_TEST_FUTURE,
                             ERA5_DIR,
                             CMIP6_RAW_DIR,
                             DATASET_DIR,
                             DATASET_BC_DIR,
                             CONFIG)

def get_era5_dataset(date):
    file = glob.glob(str(ERA5_DIR/f'tas*_{date.year}_*'))[0]
    ds = xr.open_dataset(file)
    ds = ds.sel(time=ds.time.dt.date == date).isel(time=0)
    ds = standardize_dims_and_coords(ds)
    ds = standardize_longitudes(ds)
    ds = ds.reindex(lat=ds.lat[::-1])
    ds = crop_domain_from_ds(ds, CONFIG['eobs']['domain']['europe'])
    return ds

def get_cmip6_dataset(date):
    if date < pd.Timestamp('2015-01-01').date():
        file = glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*historical*r1i1p1f2*'))[0]
    else:
        file = np.sort(glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*ssp585*r1i1p1f2*')))[0]
    ds = xr.open_dataset(file)
    ds = ds.sel(time=ds.time.dt.date == date).isel(time=0)
    ds = standardize_longitudes(ds)
    ds = crop_domain_from_ds(ds, CONFIG['eobs']['domain']['europe'])
    return ds

if __name__=='__main__':
    
    #### TRAIN HISTORIQUE DATASET
    era5_train_hist = []
    cmip6_train_hist = []
    for i, date in enumerate(DATES_BC_TRAIN_HIST):
        print(date)
        ds_cmip6 = get_cmip6_dataset(date.date()) # 1er membre
        if i == 0:
            lon, lat = ds_cmip6.lon.values, ds_cmip6.lat.values
            coordinates = {'lon': lon,
                           'lat': lat}
            np.savez(DATASET_BC_DIR/f'coordinates.npz', **coordinates)
        ds_era5 = get_era5_dataset(date.date())
        ds_era5_to_cmip6 = interpolation_target_grid(ds_era5, ds_target=ds_cmip6, method="conservative_normed") # tout à la résolution cmip6
        tas_era5 = ds_era5_to_cmip6.tas.values
        tas_cmip6 = ds_cmip6.tas.values
        era5_train_hist.append(tas_era5)
        cmip6_train_hist.append(tas_cmip6)
    era5_train_hist = np.stack(era5_train_hist, axis = 0)
    cmip6_train_hist = np.stack(cmip6_train_hist, axis = 0)
    train_hist = {'era5' : era5_train_hist,
                'cmip6' : cmip6_train_hist,
                'dates': DATES_BC_TRAIN_HIST}
    np.savez(DATASET_BC_DIR/f'bc_train_hist.npz', **train_hist)
    

    #### TEST HISTORIQUE DATASET
    era5_test_hist = []
    cmip6_test_hist = []
    for date in DATES_BC_TEST_HIST:    
        print(date)
        ds_cmip6 = get_cmip6_dataset(date.date()) # 1er membre
        ds_era5 = get_era5_dataset(date.date())
        ds_era5_to_cmip6 = interpolation_target_grid(ds_era5, ds_target=ds_cmip6, method="conservative_normed") # tout à la résolution cmip6
        tas_era5 = ds_era5_to_cmip6.tas.values
        tas_cmip6 = ds_cmip6.tas.values
        era5_test_hist.append(tas_era5)
        cmip6_test_hist.append(tas_cmip6)
    era5_test_hist = np.stack(era5_test_hist, axis = 0)
    cmip6_test_hist = np.stack(cmip6_test_hist, axis = 0)
    test_hist = {'era5' : era5_test_hist,
                'cmip6' : cmip6_test_hist,
                'dates': DATES_BC_TEST_HIST}
    np.savez(DATASET_BC_DIR/f'bc_test_hist.npz', **test_hist)
    

    #### TEST FUTUR DATASET
    cmip6_test_future = []
    for date in DATES_BC_TEST_FUTURE:    
        print(date)
        ds_cmip6 = get_cmip6_dataset(date.date())
        tas_cmip6 = ds_cmip6.tas.values
        cmip6_test_future.append(tas_cmip6)
    cmip6_test_future = np.stack(cmip6_test_future, axis = 0)
    test_future = {'cmip6' : cmip6_test_future,
                'dates' : DATES_BC_TEST_FUTURE}
    np.savez(DATASET_BC_DIR/f'bc_test_future.npz', **test_future)
        
   