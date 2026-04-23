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
from iriscc.datautils import Data, interpolation_target_grid, crop_domain_from_ds, return_unit

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
        # 1. Prediction variables (Predictors)
        x = []
        for var in CONFIG[self.exp]['input_vars']:
            if var == 'elevation':
                # Skip re-getting elevation as a dynamic variable if it's static
                continue
            # Native-First acquisition via modular plugins
            ds = self.get_era5_dataset(var, date)
            data = ds[var].values
            x.append(data)
        x = np.stack(x, axis=0)

        # 2. Target variable
        y_ds = self.get_target_dataset(self.target, 'tas', date)
        y = y_ds['tas'].values

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
