'''
Build dataset for training purpuse in Phase 1.
This script processes input data (ERA5) and target data (ex : SAFRAN or EOBS) for a given experiment.

date : 16/07/2025
author : Zoé GARCIA
'''

import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import argparse
import datetime
from typing import Tuple

from iriscc.plotutils import plot_test
from iriscc.datautils import (standardize_dims_and_coords, 
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
                     baseline: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if baseline:
            y_hat = self.baseline_data(date)
            y =self.target_data(date)
            sample = {'y_hat': y_hat,
                      'y': y}
            dataset = DATASET_DIR / f'dataset_{self.exp}_baseline'
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

        for var in self.input_vars:
            if var == 'elevation':
                ds = xr.open_dataset(self.orog_file)
                ds = crop_domain_from_ds(standardize_dims_and_coords(ds), self.domain)
            else:
                ds_era5 = get_data.get_era5_dataset(var, date)
                ds_gcm = get_data.get_gcm_dataset('tas', date, self.ssp) # default value
                ds_era5_to_gcm = interpolation_target_grid(ds_era5, 
                                                        ds_target=ds_gcm, 
                                                        method="conservative_normed")
                ds = reformat_as_target(ds_era5_to_gcm, 
                                        target_file=self.target_file, 
                                        method='conservative_normed', 
                                        domain=self.domain, 
                                        crop_target=True, mask=True)
            data = ds[var].values
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
        y_hat = np.stack(y_hat, axis=0)
        return y_hat

    

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Build dataset for experiment ")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')
    parser.add_argument('--plot', action='store_true', help='Plot the data', default=False)
    parser.add_argument('--baseline', action='store_true', help='Use baseline data instead of input data', default=False) 
    args = parser.parse_args()
    exp = args.exp

    dataset_builder = DatasetBuilder(exp)
    for i, date in enumerate(DATES):
        x, y = dataset_builder.process_date(date, 
                                            plot=args.plot, 
                                            baseline=args.baseline)

        

    
