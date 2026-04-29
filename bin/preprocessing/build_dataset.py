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
                             CONFIG,
                             ERA5_OROG_FILE)
from iriscc.datautils import (Data, 
                              interpolation_target_grid, 
                              crop_domain_from_ds, 
                              return_unit, 
                              reformat_as_target,
                              standardize_longitudes)

# Standard temporal axis for 1980 calibration
DATES = pd.date_range('1980-01-01', '2014-12-31', freq='D')
ERA5_BRIDGE_DOMAIN_MARGIN = 0.5

class DatasetBuilder(Data):
    """
    Stabilized Reconstruction Engine (v86.74)
    Handles the reconstruction of high-resolution predictors via modular handlers.
    Fixed for 3D parity and Conservative Interpolation.
    [PARITY-BROKEN]: Temporal aggregation (isel vs mean) and regridding methods 
    in this version are inconsistent with the archival exp5 certification.
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
        self.orog_file = CONFIG[exp]['orog_file']
        self.ssp = CONFIG[exp]['ssp']
        
        # v86.74 Archival Sync: Load and stabilize Target Orography once
        self.ds_target = xr.open_dataset(str(self.target_file), engine='netcdf4')
        if 'time' in self.ds_target.dims:
            self.ds_target = self.ds_target.isel(time=0)
        self.ds_target_orog = xr.open_dataset(str(self.orog_file), engine='netcdf4')
        if 'time' in self.ds_target_orog.dims:
            self.ds_target_orog = self.ds_target_orog.isel(time=0)
        if 'longitude' in self.ds_target_orog.coords: 
            self.ds_target_orog = self.ds_target_orog.rename({'longitude':'lon', 'latitude':'lat'})
        self.ds_target_orog = standardize_longitudes(self.ds_target_orog)
        
        # Ensure destination exists
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        super().__init__(domain=self.domain)

    def process_date(self, date, plot=False, baseline=False):
        """
        Build dataset for training purpose in Phase 1.
        [PARITY-BROKEN]: This version produces a 10 K spatial drift against exp5.
        DANGER: Missing the intermediate GCM interpolation bridge used in production.
        """
        # --- 1. TARGET ACQUISITION (Archival Target: South-at-Top) ---
        # Raw EOBS is South-at-Top. e7-Anchor baseline uses Unflipped orientation.
        ds_target = self.get_target_dataset(self.target, 'tas', date)
        # Restore Archival Universal Orientation: South-at-Top
        if ds_target.lat[0] > ds_target.lat[-1]:
            ds_target = ds_target.reindex(lat=ds_target.lat[::-1])
        
        y = ds_target['tas'].values
        if y.ndim == 2: y = np.expand_dims(y, axis=0)
        
        # --- 2. PREDICTOR ACQUISITION (a7fe74c Anchor Path) ---
        x = []
        for var in CONFIG[self.exp]['input_vars']:
            if var == 'elevation':
                # Anchor Path: Manual selection on STR-hardened path
                ds_elev = xr.open_dataset(str(CONFIG[self.exp]['orog_file']), engine='netcdf4')
                if 'time' in ds_elev.dims: ds_elev = ds_elev.isel(time=0)
                
                # Align Orientation
                if ds_elev.lat[0] > ds_elev.lat[-1]: ds_elev = ds_elev.reindex(lat=ds_elev.lat[::-1])
                
                data = ds_elev['elevation'].values if 'elevation' in ds_elev else ds_elev['z'].values
                # MANDATORY: Mask Alignment Protocol
                data = np.where(np.isnan(y[0]), np.nan, data)
                x.append(data)
            else:
                # --- ARCHIVAL RECONSTRUCTION: THE GCM BRIDGE ---
                # Step 1: Acquire Raw ERA5
                bridge_domain = [
                    self.domain[0] - ERA5_BRIDGE_DOMAIN_MARGIN,
                    self.domain[1] + ERA5_BRIDGE_DOMAIN_MARGIN,
                    self.domain[2] - ERA5_BRIDGE_DOMAIN_MARGIN,
                    self.domain[3] + ERA5_BRIDGE_DOMAIN_MARGIN,
                ]
                ds_era5 = Data(domain=bridge_domain).get_era5_dataset(var, date)
                
                # Step 2: Intermediate Smoothing (Regrid to GCM Reference)
                # This replicates the archival low-pass filtering.
                ds_gcm_ref = self.get_gcm_dataset('tas', date, self.ssp)
                ds_era5_to_gcm = interpolation_target_grid(ds_era5, 
                                                           ds_target=ds_gcm_ref, 
                                                           method="conservative_normed",
                                                           reuse_weights=True)
                
                # Step 3: Final Downscaling to EOBS Target
                ds = reformat_as_target(ds_era5_to_gcm, 
                                        target_file=self.target_file, 
                                        method='conservative_normed', 
                                        domain=self.domain, 
                                        crop_target=True,
                                        mask=True) # Archival mask=True enabled
                
                # Extract DataArray
                var_name = var if var in ds.data_vars else list(ds.data_vars)[0]
                data = ds[var_name].values
                
                # Align predictor NaNs to the certified target footprint.
                data = np.where(np.isnan(y[0]), np.nan, data)
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
            # v86.74 Standard: Save to CONFIG[exp]['dataset'] if output_dir not specified
            output_dir = dataset_builder.dataset
            output_path = Path(output_dir) / f"sample_{date.strftime('%Y%m%d')}.npz"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.savez(output_path, x=x, y=y)
            # print(f"--- Snapshot Saved: {output_path} ---", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to process {date.date()}: {e}", flush=True)
            raise e
