"""
Build dataset of corrected GCM data for network inference. 
This step is also performed by bias_correction_ibicus.py but can be usefull 
when an issue appears and don't want to run the whole bias correction process again.

date : 16/07/2025
author : Zoé GARCIA
"""


import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import argparse

from iriscc.settings import (DATES_BC_TEST_HIST, 
                             CONFIG,
                             DATASET_BC_DIR,
                             CONFIG,
                             GCM_RAW_DIR,
                             DATES_BC_TRAIN_HIST, 
                             DATES_BC_TEST_FUTURE)
from iriscc.datautils import reformat_as_target, Data

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Build dataset of corrected GCM data")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp5)', default='exp5')
    parser.add_argument('--var', type=str, help='Variable to use', default='tas')
    args = parser.parse_args()

    exp = args.exp
    var = args.var
    ssp = CONFIG[exp]['ssp']   
    domain = CONFIG[exp]['domain']
    orog_file = CONFIG[exp]['orog_file']
    target_file = CONFIG[exp]['target_file']
    dataset = CONFIG[exp]['dataset']

    get_data = Data(domain=domain)


    train = GCM_RAW_DIR/f'CNRM-CM6-1-BC/{var}_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc'
    ds_train_hist_bc = xr.open_dataset(train)
    test = GCM_RAW_DIR/f'CNRM-CM6-1-BC/{var}_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc'
    ds_test_hist_bc = xr.open_dataset(test)
    test_futur = GCM_RAW_DIR/f'CNRM-CM6-1-BC/{var}_day_CNRM-CM6-1_{ssp}_r1i1p1f2_gr_20150101-21001231_bc.nc'
    ds_test_future_bc = xr.open_dataset(test_futur)

    for date in DATES_BC_TRAIN_HIST:
        print(date)
        x = []

        ds = xr.open_dataset(orog_file)
        x.append(ds['elevation'].values)
        ds_train_hist_bc_i = ds_train_hist_bc.sel(time=ds_train_hist_bc.time.dt.date == date.date())
        ds_train_hist_bc_i = ds_train_hist_bc_i.isel(time=0, drop=True)

        ds_train_hist_bc_i = reformat_as_target(ds_train_hist_bc_i, 
                                            target_file=target_file,
                                            domain=domain, 
                                            method="conservative_normed",
                                            mask=True
                                            )
        


        x.append(ds_train_hist_bc_i.tas.values)
        x = np.stack(x, axis = 0)
        y = get_data.get_target_dataset(target = CONFIG[exp]['target'], 
                                                var = var, 
                                                date=date)
        sample = {'x' : x,
                    'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_BC_DIR/f'dataset_{exp}_test_gcm_bc/sample_{date_str}.npz', **sample)



    for date in DATES_BC_TEST_HIST:
        print(date)
        x = []

        ds = xr.open_dataset(orog_file)
        x.append(ds['elevation'].values)

        ds_test_hist_bc_i = ds_test_hist_bc.sel(time=ds_test_hist_bc.time.dt.date == date.date())
        ds_test_hist_bc_i = ds_test_hist_bc_i.isel(time=0, drop=True)

        ds_test_hist_bc_i = reformat_as_target(ds_test_hist_bc_i, 
                                            target_file=target_file,
                                            domain=domain, 
                                            method="conservative_normed",
                                            mask=True)
                                            

        x.append(ds_test_hist_bc_i.tas.values)
        x = np.stack(x, axis = 0)
        y = get_data.get_target_dataset(target = CONFIG[exp]['target'], 
                                                var = var, 
                                                date=date)    
        sample = {'x' : x,
                    'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_BC_DIR/f'dataset_{exp}_test_gcm_bc/sample_{date_str}.npz', **sample)



    for date in DATES_BC_TEST_FUTURE:
        print(date)
        x = []

        ds = xr.open_dataset(orog_file)
        x.append(ds['elevation'].values)

        ds_test_future_bc_i = ds_test_future_bc.sel(time=ds_test_future_bc.time.dt.date == date.date())
        ds_test_future_bc_i = ds_test_future_bc_i.isel(time=0, drop=True)
        ds_test_future_bc_i = reformat_as_target(ds_test_future_bc_i, 
                                            target_file=target_file,
                                            domain=domain, 
                                            method="conservative_normed",
                                            mask=True)

        x.append(ds_test_future_bc_i.tas.values)
        x = np.stack(x, axis = 0)
        
        sample = {'x' : x}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_BC_DIR/f'dataset_{exp}_test_gcm_bc/sample_{date_str}.npz', **sample)
