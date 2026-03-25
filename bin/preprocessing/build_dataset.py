'''
Build dataset for training purposes in Phase 1.

This script processes input data (ERA5) and target data (e.g., SAFRAN or EOBS) for a given experiment.

date : 16/07/2025
author : Zoé GARCIA
'''

import argparse
import datetime
import sys
from typing import Tuple

sys.path.append('.')

import numpy as np
import xarray as xr

from iriscc.datautils import (Data, crop_domain_from_ds,
                             interpolation_target_grid, reformat_as_target,
                             standardize_dims_and_coords)
from iriscc.plotutils import plot_test
from iriscc.settings import CONFIG, DATASET_DIR, DATES, GRAPHS_DIR

   
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
    def __init__(self, exp:str):
        self.exp = exp
        self.dataset = CONFIG[exp]['dataset']
        self.domain = CONFIG[exp]['domain']
        self.target = CONFIG[exp]['target']
        self.target_vars = CONFIG[exp]['target_vars']
        self.input_vars = CONFIG[exp]['input_vars']
        self.target_file = CONFIG[exp]['target_file']
        self.orog_file = CONFIG[exp]['orog_file']
        self.ssp = CONFIG[exp]['ssp']
    
    def process_date(self, 
                     date: datetime.date, 
                     plot: bool = False, 
                     baseline: bool = False,
                     force: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        date_str = date.date().strftime('%Y%m%d')
        if baseline:
            dataset = DATASET_DIR / f'dataset_{self.exp}_baseline'
        else:
            dataset = self.dataset
        dataset.mkdir(parents=True, exist_ok=True)
        
        target_path = dataset / f'sample_{date_str}.npz'
        if target_path.exists() and not plot and not force:
            # print(f"Skipping existing sample: {target_path}", flush=True)
            return None, None

        if baseline:
            x = self.baseline_data(date)
            y = self.target_data(date)
            sample = {'x': x,
                      'y': y}
        else:
            x = self.input_data(date)
            y = self.target_data(date)
            sample = {'x': x, 
                        'y': y}

        if plot:
            plot_test(x[1], 'Input ERA5', GRAPHS_DIR/'test.png')
        else:
            np.savez(target_path, **sample)
        return x, y

    def input_data(self, 
                   date: datetime.date) -> np.ndarray:
        x = []
        get_data = Data(self.domain)
        input_source = CONFIG[self.exp].get('input_source', 'era5')

        for var in self.input_vars:
            if var == 'elevation':
                ds = xr.open_dataset(self.orog_file)
                ds = crop_domain_from_ds(standardize_dims_and_coords(ds), self.domain)
            else:
                # Load input dataset (ERA5, CERRA, etc.)
                if input_source == 'era5':
                    ds_input = get_data.get_era5_dataset(var, date)
                elif input_source == 'cerra':
                    ds_input = get_data.get_cerra_dataset(var, date)
                else:
                    # Fallback to general target loading if needed for demonstrators
                    ds_input = get_data.get_target_dataset(target=input_source, var=var, date=date)

                ds_gcm = get_data.get_gcm_dataset('tas', date, self.ssp) 
                ds_input_to_gcm = interpolation_target_grid(ds_input, 
                                                           ds_target=ds_gcm, 
                                                           method="conservative_normed")
                ds = reformat_as_target(ds_input_to_gcm, 
                                        target_file=self.target_file, 
                                        method='conservative_normed', 
                                        domain=self.domain, 
                                        crop_target=True, mask=True)
            data = ds[var].values
            x.append(data)
        return np.stack(x, axis=0)

    def target_data(self, 
                    date: datetime.date) -> np.ndarray:
        get_data = Data(self.domain)
        y = []
        for var in self.target_vars:
            ds = get_data.get_target_dataset(target=self.target, var=var, date=date, exp=self.exp)
            data = ds[var].values
            y.append(data)
        return np.stack(y, axis=0)
    
    def baseline_data(self, 
                      date: datetime.date) -> np.ndarray:
        get_data = Data(self.domain)
        y_hat = []
        for var in self.target_vars:
            ds_era5 = get_data.get_era5_dataset(var, date)
            ds_gcm = get_data.get_gcm_dataset(var, date, self.ssp)
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
        return np.stack(y_hat, axis=0)

    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        msg = 'Boolean value expected.'
        raise argparse.ArgumentTypeError(msg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Build dataset for experiment ")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')

    parser.add_argument(
        '--plot', type=str2bool, nargs='?', const=True,
        help='Plot the data', default=False
    )
    parser.add_argument(
        '--baseline', type=str2bool, nargs='?', const=True,
        help='Use baseline data instead of input data', default=False
    )
    parser.add_argument(
        '--force', type=str2bool, nargs='?', const=True,
        help='Force data regeneration', default=False
    )
    args = parser.parse_args()
    exp = args.exp

    dataset_builder = DatasetBuilder(exp)
    
    # Ensure dataset directories exist
    dataset_builder.dataset.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / f'dataset_{exp}_baseline').mkdir(parents=True, exist_ok=True)

    total = len(DATES)
    for i, date in enumerate(DATES):
        print(f"[{datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')}] Processing date {date.date()} ({i+1}/{total})", flush=True)
        x, y = dataset_builder.process_date(date, 
                                            plot=args.plot, 
                                            baseline=args.baseline,
                                            force=args.force)

        

    
