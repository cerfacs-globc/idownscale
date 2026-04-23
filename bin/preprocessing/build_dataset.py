'''
Build dataset for training purpuse in Phase 1.
This script processes input data (ERA5) and target data (ex : SAFRAN or EOBS) for a given experiment.

date : 16/07/2025
author : Zoé GARCIA
'''

import sys
import os
sys.path.append('.')

import xarray as xr
import numpy as np
import argparse
import datetime
from pathlib import Path
import pandas as pd
from typing import Tuple

import iriscc

from iriscc.plotutils import plot_test
from iriscc.datautils import (standardize_era5_geometry, standardize_gcm_geometry,
                              standardize_eobs_geometry, standardize_longitudes,
                              interpolation_target_grid, 
                              reformat_as_target,
                              crop_domain_from_ds,
                              Data)
from iriscc.settings import (DATES,
                             CONFIG,
                             GRAPHS_DIR,
                             DATASET_DIR)

   
class DatasetBuilder:
    """
    DatasetBuilder is a class responsible for building datasets for a given experiment.
    It processes input and target data based on specified configurations and dates.
    Attributes:
        exp (str): The experiment identifier.
        dataset (str): The dataset path for the experiment.
        domain (str): The domain for the dataset.
        target (str): The target type (e.g., 'safran', 'eobs').
        target_vars (list): List of target variable names.
        input_vars (list): List of input variable names.
        target_file (str): Path to the target file.
        orog_file (str): Path to the orography file.
        ssp (str): Shared socioeconomic pathway identifier.
    Methods:
        process_date(date: datetime.date, plot: bool = False, baseline: bool = False) -> Tuple[np.ndarray, np.ndarray]:
            Processes the data for a specific date, optionally plotting or using baseline data.
        input_data(date: datetime.date) -> np.ndarray:
            Retrieves and formats input data for the specified date [C,H,W].
        target_data(date: datetime.date) -> np.ndarray:
            Retrieves and formats target data for the specified date [C,H,W].
        baseline_data(date: datetime.date) -> np.ndarray:
            Retrieves and formats baseline input data for the specified date [C,H,W]
    """
    def __init__(self, exp:str, args):
        self.exp = exp
        self.audit_dir = Path(args.audit_dir) if args.audit_dir else None
        self.dataset = CONFIG[exp]['dataset']
        self.domain = CONFIG[exp]['domain']
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        
        #### TRAIN HISTORIQUE DATASET
        self.target = CONFIG[exp]['target']
        self.target_vars = CONFIG[exp]['target_vars']
        self.input_vars = CONFIG[exp]['input_vars']
        self.target_file = CONFIG[exp]['target_file']
        self.orog_file = CONFIG[exp]['orog_file']
        self.ssp = CONFIG[exp]['ssp']
        
        # Ensure dataset directory exists
        self.dataset = self.audit_dir if self.audit_dir else Path(CONFIG[exp]['dataset'])
        os.makedirs(self.dataset, exist_ok=True)
    
    def process_date(self, 
                     date: datetime.date, 
                     plot: bool = False, 
                     baseline: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if baseline:
            y_hat = self.baseline_data(date)
            y =self.target_data(date)
            sample = {'y_hat': y_hat,
                      'y': y}
            dataset = self.dataset
        else:
            x = self.input_data(date)
            y = self.target_data(date)
            sample = {'x': x, 
                        'y': y}
            dataset = self.dataset
        date_str = date.date().strftime('%Y%m%d')
        if plot:
            plot_test(x[1], 'Input ERA5', GRAPHS_DIR/'test.png')
        else:
            np.savez(dataset / f'sample_{date_str}.npz', **sample)
        return x, y

    def input_data(self, 
                   date: datetime.date) -> np.ndarray:
        x = []
        get_data = Data(self.domain)

        # Performance fix: load GCM once per day (always 'tas'), not once per variable
        ds_gcm = get_data.get_gcm_dataset('tas', date, self.ssp,
                                          lapse_rate_correction=True,
                                          orog_target_file=self.orog_file,
                                          reuse_weights=True)

        for var in self.input_vars:
            if var == 'elevation':
                ds = xr.open_dataset(self.orog_file)
                ds = crop_domain_from_ds(standardize_eobs_geometry(ds), self.domain)
            else:
                ds_era5 = get_data.get_era5_dataset(var, date,
                                                   lapse_rate_correction=True,
                                                   orog_target_file=self.orog_file,
                                                   reuse_weights=True)
                ds_era5_to_gcm = interpolation_target_grid(ds_era5, 
                                                        ds_target=ds_gcm, 
                                                        method="bilinear")
                ds = reformat_as_target(ds_era5_to_gcm, 
                                        target_file=self.target_file, 
                                        method='bilinear', 
                                        domain=self.domain, 
                                        crop_target=True, mask=False,
                                        reuse_weights=True)
            data = ds[var].values.astype(np.float64)
            x.append(data)
        x = np.stack(x, axis=0)
        return x

    def target_data(self, 
                    date: datetime.date) -> np.ndarray:
        get_data = Data(self.domain)
        y = []
        if self.target == 'safran':
            for var in self.target_vars:
                ds = get_data.get_safran_dataset(var, date)
                data = ds[var].values
                y.append(data)
        elif self.target == 'eobs':
            for var in self.target_vars:
                ds = get_data.get_eobs_dataset(var, date)
                data = ds[var].values
                y.append(data)
        y = np.stack(y, axis=0)
        return y
    
    def baseline_data(self, 
                      date: datetime.date) -> np.ndarray:
        get_data = Data(self.domain)
        y_hat = []
        for var in self.input_vars:
            ds_era5 = get_data.get_era5_dataset(var, date,
                                               lapse_rate_correction=True,
                                               orog_target_file=self.orog_file)
            ds_gcm = get_data.get_gcm_dataset(var, date, self.ssp,
                                              lapse_rate_correction=True,
                                              orog_target_file=self.orog_file)
            ds_era5_to_gcm = interpolation_target_grid(ds_era5, 
                                                       ds_target=ds_gcm, 
                                                       method="conservative_normed")
            ds = reformat_as_target(ds_era5_to_gcm, 
                                    target_file=self.target_file, 
                                    method='bilinear', 
                                    domain=self.domain, 
                                    crop_target=True, mask=True)
            data = ds[var].values
            y_hat.append(data)
        y_hat = np.stack(y_hat, axis=0)
        return y_hat

    

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Build dataset for experiment ")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')
    parser.add_argument('--plot', action='store_true', help='Plot the data', default=False)
    parser.add_argument('--baseline', action='store_true', help='Use baseline data instead of input data', default=False)
    parser.add_argument('--start_date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--audit_dir', type=str, default=None, help='Directory to save audit .npz samples')
    args = parser.parse_args()
    exp = args.exp

    if args.start_date and args.end_date:
        start = pd.Timestamp(args.start_date)
        end = pd.Timestamp(args.end_date)
        DATES = [d for d in DATES if start <= d <= end]
        print(f"[WINDOWING] Processing {len(DATES)} dates from {start.date()} to {end.date()}")

    if args.audit_dir and not os.path.exists(args.audit_dir):
        os.makedirs(args.audit_dir, exist_ok=True)

    dataset_builder = DatasetBuilder(exp=args.exp, args=args)
    for i, date in enumerate(DATES):
        print(f"[PROCESS] {date.date()}")
        x, y = dataset_builder.process_date(date, 
                                            plot=args.plot, 
                                            baseline=args.baseline)
        
        if args.audit_dir:
            audit_path = os.path.join(args.audit_dir, f"sample_{date.strftime('%Y%m%d')}.npz")
            np.savez(audit_path, x=x, y=y)
            print(f"--- Certified Audit Snapshot Saved: {audit_path} ---")

        

    
