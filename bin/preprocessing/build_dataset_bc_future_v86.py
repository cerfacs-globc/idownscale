import sys
sys.path.append('.')

from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from bin.preprocessing.build_dataset import Data
from iriscc.datautils import interpolation_target_grid
from iriscc.settings import (DATES_BC_TEST_FUTURE,
                             DATASET_BC_DIR,
                             CONFIG)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Future-Only Synthesis Catch-up Engine")
    parser.add_argument('--simu', type=str, default='gcm')
    parser.add_argument('--ssp', type=str, default='ssp585')
    parser.add_argument('--var', type=str, default='tas') 
    parser.add_argument('--exp', type=str, default='exp5')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    domain = CONFIG[args.exp]['domain']
    get_data = Data(domain=domain)
    
    base_dir = Path(args.output_dir) if args.output_dir else DATASET_BC_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    
    label = "test_future"
    dates = DATES_BC_TEST_FUTURE
    
    print(f"--- Processing {label} Period (2015-2100) ---", flush=True)
    
    era5_list = []
    simu_list = []
    for date in dates:
        print(f"[{label}] {date.date()}", flush=True)
        ds_era5 = get_data.get_era5_dataset(args.var, date,
                                           lapse_rate_correction=CONFIG[args.exp].get('lapse_rate_correction', False),
                                           orog_target_file=CONFIG[args.exp].get('orog_file'))

        ds_simu = get_data.get_gcm_dataset(args.var, date, ssp=args.ssp)
        ds_era5_to_gcm = interpolation_target_grid(ds_era5, ds_target=ds_simu, method="conservative_normed")
                                                   
        era5_list.append(ds_era5_to_gcm[args.var].values)
        simu_list.append(ds_simu[args.var].values)
    
    # Persistence
    era5_stack = np.stack(era5_list, axis=0)
    simu_stack = np.stack(simu_list, axis=0)
    output_path = base_dir / f'bc_{label}_{args.simu}.npz'
    np.savez(output_path, era5=era5_stack, **{args.simu: simu_stack}, dates=dates)
    print(f"--- {label} Saved to {output_path} ---", flush=True)
