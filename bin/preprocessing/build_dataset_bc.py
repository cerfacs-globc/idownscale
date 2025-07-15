''' Data preprocessing for bias correction, for RCM or GCM data. 
NPZ files are saved with predictors (simu), predictants (ERA5) and dates for training, validation and test
'''

import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import pandas as pd
import glob
import argparse

from bin.preprocessing.build_dataset import Data
from iriscc.plotutils import plot_test
from iriscc.datautils import (
                              interpolation_target_grid, 
                              crop_domain_from_ds,)
from iriscc.settings import (DATES_TEST,
                             DATES_BC_TRAIN_HIST,
                             DATES_BC_TEST_HIST,
                             DATES_BC_TEST_FUTURE,
                             DATASET_DIR,
                             DATASET_BC_DIR,
                             CONFIG,
                             GRAPHS_DIR,
                             ALADIN_PROJ_PYPROJ)



if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Build dataset for bias correction ")
    parser.add_argument('--simu', type=str, help='simu or rcm', default='simu')
    parser.add_argument('--ssp', type=str, help='ssp585 or historical', default='ssp585')
    parser.add_argument('--var', type=str, help='variable to use', default='tas') 
    args = parser.parse_args()

    domain = [-12.5, 27.5, 31., 71.]
    get_data = Data(domain=domain)
    
    '''
    #### TRAIN HISTORIQUE DATASET
    era5_train_hist = []
    simu_train_hist = []
    for i, date in enumerate(DATES_BC_TRAIN_HIST):
        print(date)
        if args.simu == 'gcm':
            ds_simu = get_data.get_gcm_dataset(args.var, date) # 1er membre
        elif args.simu == 'rcm':
            ds_simu = get_data.get_rcm_dataset(args.var, date)
            ds_gcm = get_data.get_gcm_dataset(args.var, date)
            ds = interpolation_target_grid(ds, 
                                   ds_target=crop_domain_from_ds(ds_gcm, domain), 
                                   method="conservative_normed")
        ds_era5 = get_data.get_era5_dataset(args.var, date)
        ds_era5_to_simu = interpolation_target_grid(ds_era5, ds_target=ds_simu, method="conservative_normed") # tout à la résolution simu
        tas_era5 = ds_era5_to_simu.tas.values
        tas_simu = ds_simu.tas.values
        era5_train_hist.append(tas_era5)
        simu_train_hist.append(tas_simu)
    era5_train_hist = np.stack(era5_train_hist, axis = 0)
    simu_train_hist = np.stack(simu_train_hist, axis = 0)
    train_hist = {'era5' : era5_train_hist,
                args.simu : simu_train_hist,
                'dates': DATES_BC_TRAIN_HIST}
    np.savez(DATASET_BC_DIR/f'bc_train_hist_{args.simu}.npz', **train_hist)
    

    
    #### TEST HISTORIQUE DATASET
    era5_test_hist = []
    simu_test_hist = []
    for date in DATES_BC_TEST_HIST:    
        print(date)
        if args.simu == 'gcm':
            ds_simu = get_data.get_gcm_dataset(args.var, date)
        else :
            ds_simu = get_data.get_rcm_dataset(args.var, date)
            ds_gcm = get_data.get_gcm_dataset(args.var, date)
            ds = interpolation_target_grid(ds, 
                                   ds_target=crop_domain_from_ds(ds_gcm, domain), 
                                   method="conservative_normed")
        ds_era5 = get_data.get_era5_dataset(args.var, date)
        ds_era5_to_simu = interpolation_target_grid(ds_era5, ds_target=ds_simu, method="conservative_normed") # tout à la résolution simu
        tas_era5 = ds_era5_to_simu.tas.values
        tas_simu = ds_simu.tas.values
        era5_test_hist.append(tas_era5)
        simu_test_hist.append(tas_simu)
    era5_test_hist = np.stack(era5_test_hist, axis = 0)
    simu_test_hist = np.stack(simu_test_hist, axis = 0)
    test_hist = {'era5' : era5_test_hist,
                args.simu : simu_test_hist,
                'dates': DATES_BC_TEST_HIST}
    np.savez(DATASET_BC_DIR/f'bc_test_hist_{args.simu}.npz', **test_hist)
    '''

    #### TEST FUTUR DATASET
    simu_test_future = []
    for date in DATES_BC_TEST_FUTURE:    
        print(date)
        if args.simu == 'gcm':
            ds_simu = get_data.get_gcm_dataset(args.var, date, args.ssp) # 1er membre
        else :
            ds_simu = get_data.get_rcm_dataset(args.var, date, args.ssp)
            ds_gcm = get_data.get_gcm_dataset(args.var, date=date, ssp=args.ssp)
            ds_simu = interpolation_target_grid(ds_simu, 
                                   ds_target=ds_gcm, 
                                   method="conservative_normed",
                                   bounds_method="2")
        tas_simu = ds_simu.tas.values
        ds_simu.close()
        simu_test_future.append(tas_simu)
    simu_test_future = np.stack(simu_test_future, axis = 0)
    test_future = {args.simu : simu_test_future,
                'dates' : DATES_BC_TEST_FUTURE}
    np.savez(DATASET_BC_DIR/f'bc_test_future_{args.simu}.npz', **test_future)
