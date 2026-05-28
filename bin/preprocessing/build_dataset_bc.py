import sys
sys.path.append('.')

from pathlib import Path
import os
import numpy as np
import pandas as pd
import argparse

from bin.preprocessing.build_dataset import Data
from iriscc.datautils import (interpolation_target_grid, 
                               crop_domain_from_ds)
from iriscc.settings import (DATES_BC_TRAIN_HIST,
                             ALADIN_PROJ_PYPROJ,
                             DATES_BC_TEST_HIST,
                             DATES_BC_TEST_FUTURE,
                             DATASET_BC_DIR,
                             CONFIG,
                             get_simu_family,
                             get_simu_source)

ERA5_BC_DOMAIN_MARGIN = float(os.getenv("IDOWNSCALE_ERA5_BC_DOMAIN_MARGIN", "0.0"))

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
    bc_domain = CONFIG[args.exp].get('bc_domain', domain)
    bc_reanalysis_source = CONFIG[args.exp].get('bc_reanalysis_source', 'era5')
    gcm_source = CONFIG[args.exp].get('gcm_source', 'gcm_cnrm_cm6_1')
    simu_source = get_simu_source(args.exp, args.simu)
    simu_family = get_simu_family(args.exp, args.simu)
    get_bc_data = Data(domain=bc_domain)
    
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
            era5_domain = [
                bc_domain[0] - ERA5_BC_DOMAIN_MARGIN,
                bc_domain[1] + ERA5_BC_DOMAIN_MARGIN,
                bc_domain[2] - ERA5_BC_DOMAIN_MARGIN,
                bc_domain[3] + ERA5_BC_DOMAIN_MARGIN,
            ]
            ds_anchor = Data(domain=era5_domain).get_reanalysis_dataset(bc_reanalysis_source, args.var, DATES_BC_TRAIN_HIST[0])

        for date in dates:
            print(f"[{label}] {date.date()}", flush=True)
            
            # Acquisition Logic
            if not is_future:
                era5_domain = [
                    bc_domain[0] - ERA5_BC_DOMAIN_MARGIN,
                    bc_domain[1] + ERA5_BC_DOMAIN_MARGIN,
                    bc_domain[2] - ERA5_BC_DOMAIN_MARGIN,
                    bc_domain[3] + ERA5_BC_DOMAIN_MARGIN,
                ]
                ds_era5 = Data(domain=era5_domain).get_reanalysis_dataset(bc_reanalysis_source, args.var, date)
            else:
                ds_era5 = ds_anchor # Use the persistent anchor geometry

            if simu_family == 'gcm':
                ds_simu = get_bc_data.get_model_dataset(simu_source, args.var, date, ssp=args.ssp)
                ds_target_regrid = interpolation_target_grid(ds_era5, ds_target=ds_simu, method="conservative_normed")
            else :
                ds_simu = get_bc_data.get_model_dataset(simu_source, args.var, date, ssp=args.ssp)
                ds_gcm = get_bc_data.get_model_dataset(gcm_source, args.var, date=date, ssp=args.ssp)
                input_projection = ALADIN_PROJ_PYPROJ if simu_source == 'rcm_aladin' else None
                ds_simu = interpolation_target_grid(
                    ds_simu,
                    ds_target=crop_domain_from_ds(ds_gcm, domain),
                    method="conservative_normed",
                    input_projection=input_projection,
                )
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
    
    # Apply CLI Overrides for 31-Day Certification Benchmarking
    if args.start_date and args.end_date:
        print(f"--- Applying Benchmark Date Overrides: {args.start_date} to {args.end_date} ---", flush=True)
        DATES_BC_TRAIN_HIST = pd.date_range(start=args.start_date, end=args.end_date, freq='D')
        DATES_BC_TEST_HIST = pd.Index([])
        DATES_BC_TEST_FUTURE = pd.Index([])

    # 1. Historical Train (with reanalysis)
    if len(DATES_BC_TRAIN_HIST) > 0:
        process_period(DATES_BC_TRAIN_HIST, "train_hist")
    
    # 2. Historical Test (with reanalysis)
    if len(DATES_BC_TEST_HIST) > 0:
        process_period(DATES_BC_TEST_HIST, "test_hist")
    
    # 3. Future Projection (decoupled, no reanalysis)
    if len(DATES_BC_TEST_FUTURE) > 0:
        process_period(DATES_BC_TEST_FUTURE, "test_future")

    print(f"--- ISOLATED PRODUCTION SYNTHESIS COMPLETE ---", flush=True)
