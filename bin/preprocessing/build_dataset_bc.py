'''
Data preprocessing for bias correction, for RCM or GCM data.
NPZ files are saved with predictors (simu), predictants (ERA5) and dates for training, validation and test

date : 16/07/2025
author : Zoé GARCIA
'''
import sys
sys.path.append('.')

from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from bin.preprocessing.build_dataset import Data
from iriscc.datautils import (interpolation_target_grid, 
                              crop_domain_from_ds)
from iriscc.settings import (DATES_BC_TRAIN_HIST,
                             DATES_BC_TEST_HIST,
                             DATES_BC_TEST_FUTURE,
                             DATASET_BC_DIR,
                             CONFIG)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Build dataset for bias correction ")
    parser.add_argument('--simu', type=str, help='simu or rcm', default='simu')
    parser.add_argument('--ssp', type=str, help='ssp585 or historical', default='ssp585')
    parser.add_argument('--var', type=str, help='variable to use', default='tas') 
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp5)', default='exp5')
    parser.add_argument('--start_date', type=str, help='Start date (YYYY-MM-DD)', default=None)
    parser.add_argument('--end_date', type=str, help='End date (YYYY-MM-DD)', default=None)
    parser.add_argument('--audit_dir', type=str, help='Custom output directory', default=None)
    args = parser.parse_args()

    domain = [-12.5, 27.5, 31., 71.]
    get_data = Data(domain=domain)
    
    # Path Resolution Protocol
    base_dir = Path(args.audit_dir) if args.audit_dir else DATASET_BC_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    
    
    # 1. TRAIN HISTORIQUE DATASET
    dates_train = DATES_BC_TRAIN_HIST
    if args.start_date and args.end_date:
        dates_train = pd.date_range(start=args.start_date, end=args.end_date, freq='D')
        
    era5_train_hist = []
    simu_train_hist = []
    for i, date in enumerate(dates_train):
        print(date)
        ds_era5 = get_data.get_era5_dataset(args.var, date,
                                           lapse_rate_correction=CONFIG[args.exp].get('lapse_rate_correction', False),
                                           orog_target_file=CONFIG[args.exp].get('orog_file'))

        if args.simu == 'gcm':
            ds_simu = get_data.get_gcm_dataset(args.var, date)
            ds_era5_to_gcm = interpolation_target_grid(ds_era5, 
                                                       ds_target=ds_simu, 
                                                       method="conservative_normed")
        else :
            ds_simu = get_data.get_rcm_dataset(args.var, date)
            ds_gcm = get_data.get_gcm_dataset(args.var, date)
            ds_simu = interpolation_target_grid(ds_simu, 
                                   ds_target=crop_domain_from_ds(ds_gcm, domain), 
                                   method="conservative_normed",
                                   bounds_method="2") # method 1 doesn't work for ALADIN, don't know why
            ds_era5_to_gcm = interpolation_target_grid(ds_era5, 
                                                       ds_target=ds_gcm, 
                                                       method="conservative_normed")
        tas_era5 = ds_era5_to_gcm.tas.values
        tas_simu = ds_simu.tas.values
        era5_train_hist.append(tas_era5)
        simu_train_hist.append(tas_simu)
    era5_train_hist = np.stack(era5_train_hist, axis = 0)
    simu_train_hist = np.stack(simu_train_hist, axis = 0)
    train_hist = {'era5' : era5_train_hist,
                args.simu : simu_train_hist,
                'dates': dates_train}
    np.savez(base_dir/f'bc_train_hist_{args.simu}.npz', **train_hist)
    

    
    # 2. TEST HISTORIQUE DATASET
    dates_test = DATES_BC_TEST_HIST
    if args.start_date and args.end_date:
        dates_test = pd.date_range(start=args.start_date, end=args.end_date, freq='D')
        
    era5_test_hist = []
    simu_test_hist = []
    for date in dates_test:    
        print(date)
        ds_era5 = get_data.get_era5_dataset(args.var, date,
                                           lapse_rate_correction=CONFIG[args.exp].get('lapse_rate_correction', False),
                                           orog_target_file=CONFIG[args.exp].get('orog_file'))

        if args.simu == 'gcm':
            ds_simu = get_data.get_gcm_dataset(args.var, date)
            ds_era5_to_gcm = interpolation_target_grid(ds_era5, 
                                                       ds_target=ds_simu, 
                                                       method="conservative_normed")
        else :
            ds_simu = get_data.get_rcm_dataset(args.var, date)
            ds_gcm = get_data.get_gcm_dataset(args.var, date)
            ds_simu = interpolation_target_grid(ds_simu, 
                                   ds_target=crop_domain_from_ds(ds_gcm, domain), 
                                   method="conservative_normed",
                                   bounds_method="2")
            ds_era5_to_gcm = interpolation_target_grid(ds_era5, 
                                                       ds_target=ds_gcm, 
                                                       method="conservative_normed")

        
        tas_era5 = ds_era5_to_gcm.tas.values
        tas_simu = ds_simu.tas.values
        era5_test_hist.append(tas_era5)
        simu_test_hist.append(tas_simu)
    era5_test_hist = np.stack(era5_test_hist, axis = 0)
    simu_test_hist = np.stack(simu_test_hist, axis = 0)
    test_hist = {'era5' : era5_test_hist,
                args.simu : simu_test_hist,
                'dates': dates_test}
    np.savez(base_dir/f'bc_test_hist_{args.simu}.npz', **test_hist)
    
    print(f"STABILIZATION_COMPLETE: Phase 2 localized audit file saved to {base_dir}")
    

    #### TEST FUTUR DATASET
    if args.start_date and args.end_date:
        print("STABILIZATION_NOTICE: Skipping Future Dataset for localized audit.")
        sys.exit(0)
        
    simu_test_future = []
    for date in DATES_BC_TEST_FUTURE:    
        print(date)
        if args.simu == 'gcm':
            ds_simu = get_data.get_gcm_dataset(args.var, date, args.ssp).sortby('lat', ascending=False)
        else :
            ds_simu = get_data.get_rcm_dataset(args.var, date, args.ssp)
            ds_gcm = get_data.get_gcm_dataset(args.var, date=date, ssp=args.ssp).sortby('lat', ascending=False)
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
