import sys
import os
sys.path.append('.')

import xarray as xr
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

from iriscc.settings import (RAW_DIR, 
                             DATASET_DIR, 
                             CONFIG)
from iriscc.datautils import Data, interpolation_target_grid, crop_domain_from_ds, return_unit, reformat_as_target

# Standard temporal axis for 1980 calibration
DATES = pd.date_range('1980-01-01', '2014-12-31', freq='D')

class DatasetBuilder(Data):
    """
    Stabilized Reconstruction Engine (v86.74)
    Handles the reconstruction of high-resolution predictors via modular handlers.
    """
    def __init__(self, exp:str, args):
        self.exp = exp
        # v48 Traceability: Standardize on output_dir
        self.output_dir = Path(args.output_dir) if args.output_dir else None
        
        self.dataset = self.output_dir if self.output_dir else Path(CONFIG[exp]['dataset'])
        self.domain = CONFIG[exp]['domain']
        
        #### TARGET GEOMETRY
        self.target = CONFIG[exp]['target']
        self.target_file = CONFIG[exp]['target_file']
        
        # Ensure destination exists
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        super().__init__(domain=self.domain)

    def process_date(self, date, plot=False, baseline=False):
        """Reconstruct the predictors and target for a specific date."""
        # 1. Target variable
        target_file = CONFIG[self.exp].get('target_file')
        y_ds = xr.open_dataset(target_file)
        if 'time' in y_ds.dims:
            # v86.74 Target Alignment: Strictly select date to avoid temporal drift
            y_ds = y_ds.sel(time=str(date.date()))
        var_target = 'tas' if 'tas' in y_ds else list(y_ds.data_vars)[0]
        y = y_ds[var_target].values
        
        # 2. Prediction variables (Predictors)
        x = []
        for var in CONFIG[self.exp]['input_vars']:
            if var == 'elevation':
                msg = f"Getting {var} as static predictor with exact target mask..."
                print(msg, flush=True)
                ds_elev = xr.open_dataset(CONFIG[self.exp]['orog_file'])
                if 'time' in ds_elev.dims:
                    ds_elev = ds_elev.isel(time=0, drop=True)
                
                data = ds_elev['elevation'].values if 'elevation' in ds_elev else ds_elev['z'].values
                if data.shape != y.shape:
                    raise ValueError(f"Shape mismatch: {var} shape {data.shape} != target {y.shape}")
                
                # Enforce identical exact target mask! (1566 NaNs)
                data = np.where(np.isnan(y), np.nan, data)
                x.append(data)
            else:
                # Native-First acquisition via modular plugins
                ds = self.get_era5_dataset(var, date)
                ds = reformat_as_target(ds, target_file, method='bilinear', domain=self.domain, mask=False)
                
                var_era5 = var if var in ds.data_vars else list(ds.data_vars)[0]
                data = ds[var_era5].values
                
                if data.shape != y.shape:
                    raise ValueError(f"Shape mismatch: {var} shape {data.shape} != target {y.shape}")
                
                # Enforce identical exact target mask! (1566 NaNs)
                data = np.where(np.isnan(y), np.nan, data)
                x.append(data)
        x = np.stack(x, axis=0)

        return x, y

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Stabilized Dataset Builder")
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--baseline', action='store_true', default=False)
    parser.add_argument('--start_date', type=str, default=None)
    parser.add_argument('--end_date', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    # Temporal Windowing
    if args.start_date and args.end_date:
        start = pd.Timestamp(args.start_date)
        end = pd.Timestamp(args.end_date)
        DATES = [d for d in DATES if start <= d <= end]
        print(f"[WINDOWING] Processing {len(DATES)} dates from {start.date()} to {end.date()}", flush=True)

    # Engine Initialization
    dataset_builder = DatasetBuilder(exp=args.exp, args=args)
    
    for date in DATES:
        print(f"[PROCESS] {date.date()}", flush=True)
        try:
            x, y = dataset_builder.process_date(date, baseline=args.baseline)
            
            # Persistence Logic
            if args.output_dir:
                output_path = Path(args.output_dir) / f"sample_{date.strftime('%Y%m%d')}.npz"
                np.savez(output_path, x=x, y=y)
                print(f"--- Snapshot Saved: {output_path} ---", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to process {date.date()}: {e}", flush=True)
            raise e
