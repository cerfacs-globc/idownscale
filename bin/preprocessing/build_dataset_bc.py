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
    parser = argparse.ArgumentParser(description="Isolated Decoupled BC Dataset Builder")
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
        
        # Generic Geometry Anchor discovery for Future periods
        is_future = (label == "test_future")
        ds_anchor = None
        if is_future:
            print(f"[{label}] Generic Protocol: Establishing Geometry Anchor from {DATES_BC_TRAIN_HIST[0].date()}", flush=True)
            ds_anchor = get_data.get_era5_dataset(args.var, DATES_BC_TRAIN_HIST[0],
                                               lapse_rate_correction=CONFIG[args.exp].get('lapse_rate_correction', False),
                                               orog_target_file=CONFIG[args.exp].get('orog_file'))

        for date in dates:
            print(f"[{label}] {date.date()}", flush=True)
            
            # Acquisition Logic
            if not is_future:
                ds_era5 = get_data.get_era5_dataset(args.var, date,
                                                   lapse_rate_correction=CONFIG[args.exp].get('lapse_rate_correction', False),
                                                   orog_target_file=CONFIG[args.exp].get('orog_file'))
            else:
                ds_era5 = ds_anchor # Use the persistent anchor geometry

            if args.simu == 'gcm':
                ds_simu = get_data.get_gcm_dataset(args.var, date, ssp=args.ssp)
                ds_target_regrid = interpolation_target_grid(ds_era5, ds_target=ds_simu, method="conservative_normed")
            else :
                ds_simu = get_data.get_rcm_dataset(args.var, date, ssp=args.ssp)
                ds_gcm = get_data.get_gcm_dataset(args.var, date=date, ssp=args.ssp)
                ds_simu = interpolation_target_grid(ds_simu, ds_target=crop_domain_from_ds(ds_gcm, domain), method="conservative_normed")
                ds_target_regrid = interpolation_target_grid(ds_era5, ds_target=ds_gcm, method="conservative_normed")
            
            # Stack Logic
            if not is_future:
                era5_list.append(ds_target_regrid[args.var].values)
            
            simu_list.append(ds_simu[args.var].values)
        
        # Persistence Logic (Isolated Schema)
        simu_stack = np.stack(simu_list, axis=0)
        output_path = base_dir / f'bc_{label}_{args.simu}.npz'
        
        save_dict = {args.simu: simu_stack, 'dates': dates}
        if not is_future:
            save_dict['era5'] = np.stack(era5_list, axis=0)
            
        np.savez_compressed(output_path, **save_dict)
        print(f"--- {label} Isolated Volume Saved to {output_path} ---", flush=True)
        return simu_stack
    
    # 1. Historical Train (with reanalysis)
    process_period(DATES_BC_TRAIN_HIST, "train_hist")
    
    # 2. Historical Test (with reanalysis)
    process_period(DATES_BC_TEST_HIST, "test_hist")
    
    # 3. Future Projection (decoupled, no reanalysis)
    process_period(DATES_BC_TEST_FUTURE, "test_future")

    print(f"--- ISOLATED PRODUCTION SYNTHESIS COMPLETE ---", flush=True)
