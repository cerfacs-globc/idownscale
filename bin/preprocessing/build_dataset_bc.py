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
    parser = argparse.ArgumentParser(description="Standardized Production BC Dataset Builder")
    parser.add_argument('--simu', type=str, default='simu')
    parser.add_argument('--ssp', type=str, default='historical')
    parser.add_argument('--var', type=str, default='tas') 
    parser.add_argument('--exp', type=str, default='exp5')
    parser.add_argument('--start_date', type=str, default=None)
    parser.add_argument('--end_date', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--test', action='store_true', help='Test mode: process 1 day per period')
    args = parser.parse_args()

    domain = CONFIG[args.exp]['domain']
    get_data = Data(domain=domain)
    
    base_dir = Path(args.output_dir) if args.output_dir else DATASET_BC_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    
    def process_period(dates, label):
        print(f"--- Processing {label} Period ---", flush=True)
        if args.test:
            dates = dates[:1]
            print(f"TEST_MODE: Limiting to {dates[0].date()}", flush=True)
            
        era5_list = []
        simu_list = []
        for date in dates:
            print(f"[{label}] {date.date()}", flush=True)
            ds_era5 = get_data.get_era5_dataset(args.var, date,
                                               lapse_rate_correction=CONFIG[args.exp].get('lapse_rate_correction', False),
                                               orog_target_file=CONFIG[args.exp].get('orog_file'))

            if args.simu == 'gcm':
                ds_simu = get_data.get_gcm_dataset(args.var, date, ssp=args.ssp)
                ds_era5_to_gcm = interpolation_target_grid(ds_era5, ds_target=ds_simu, method="conservative_normed")
            else :
                ds_simu = get_data.get_rcm_dataset(args.var, date, ssp=args.ssp)
                ds_gcm = get_data.get_gcm_dataset(args.var, date=date, ssp=args.ssp)
                ds_simu = interpolation_target_grid(ds_simu, ds_target=crop_domain_from_ds(ds_gcm, domain), method="conservative_normed")
                ds_era5_to_gcm = interpolation_target_grid(ds_era5, ds_target=ds_gcm, method="conservative_normed")
                                                       
            era5_list.append(ds_era5_to_gcm[args.var].values)
            simu_list.append(ds_simu[args.var].values)
        
        # Persistence
        era5_stack = np.stack(era5_list, axis=0)
        simu_stack = np.stack(simu_list, axis=0)
        output_path = base_dir / f'bc_{label}_{args.simu}.npz'
        np.savez(output_path, era5=era5_stack, **{args.simu: simu_stack}, dates=dates)
        print(f"--- {label} Saved to {output_path} ---", flush=True)
        return era5_stack, simu_stack, dates

    # --- 1. TRAIN HISTORIQUE ---
    dates_train = DATES_BC_TRAIN_HIST
    if args.start_date and args.end_date:
        dates_train = pd.date_range(start=args.start_date, end=args.end_date, freq='D')
    
    e_train, s_train, d_train = process_period(dates_train, "train_hist")

    # Performance Guard: Skip Test/Future for localized monthly audits
    # (Disabled in --test mode to allow full pipeline validation)
    if not args.test:
        if args.start_date and args.end_date:
            print("STABILIZATION_NOTICE: Completing localized Monthly Benchmark.", flush=True)
            sys.exit(0)

    # --- 2. TEST HISTORIQUE ---
    e_test_h, s_test_h, d_test_h = process_period(DATES_BC_TEST_HIST, "test_hist")

    # --- 3. TEST FUTURE ---
    e_test_f, s_test_f, d_test_f = process_period(DATES_BC_TEST_FUTURE, "test_future")

    # --- 4. MASTER COMBINATION (v86.74 Universal) ---
    print("--- Generating Master Combined Dataset ---", flush=True)
    master_era5 = np.concatenate([e_train, e_test_h, e_test_f], axis=0)
    master_simu = np.concatenate([s_train, s_test_h, s_test_f], axis=0)
    master_dates = np.concatenate([d_train, d_test_h, d_test_f], axis=0)
    
    master_path = base_dir / f'bc_master_{args.simu}.npz'
    np.savez(master_path, era5=master_era5, **{args.simu: master_simu}, dates=master_dates)
    print(f"--- MASTER SYNTHESIS COMPLETE: {master_path} ---", flush=True)
