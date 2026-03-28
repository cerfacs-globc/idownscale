'''
Build dataset for training purposes in Phase 1.

This script processes input data (ERA5) and target data (e.g., SAFRAN or EOBS) for a given experiment.

date : 16/07/2025
author : Zoé GARCIA
'''

import argparse
import datetime
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
                    ds_input = get_data.get_era5_dataset(var, date, exp=self.exp)
                elif input_source == 'cerra':
                    ds_input = get_data.get_cerra_dataset(var, date) # Cerra not yet updated for exp
                else:
                    # Fallback to general target loading if needed for demonstrators
                    ds_input = get_data.get_target_dataset(target=input_source, var=var, date=date)

                ds_gcm = get_data.get_gcm_dataset('tas', date, self.ssp, exp=self.exp) 
                if ds_input is None or ds_gcm is None:
                    continue # Skip missing calendar dates
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
            if ds is None:
                continue # Skip missing calendar dates
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


def _process_date_worker(args):
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    exp, date_str, plot, baseline, force = args
    import datetime as _dt
    date = _dt.datetime.strptime(date_str, '%Y-%m-%d')
    builder = DatasetBuilder(exp)
    return builder.process_date(date, plot=plot, baseline=baseline, force=force)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Build dataset for experiment ")
    parser.add_argument('--exp', type=str, default='exp5', help='Experiment name (e.g., exp5)')
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
    parser.add_argument(
        '--workers', type=int, default=int(os.getenv('IDOWNSCALE_WORKERS', '1')),
        help='Number of parallel worker processes (default: IDOWNSCALE_WORKERS env or 1)'
    )
    args = parser.parse_args()
    exp = args.exp

    dataset_builder = DatasetBuilder(exp)
    dataset_builder.dataset.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / f'dataset_{exp}_baseline').mkdir(parents=True, exist_ok=True)

    total = len(DATES)
    t0 = time.monotonic()

    if args.workers <= 1:
        # Serial path (original behaviour)
        for i, date in enumerate(DATES):
            print(f"[{datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')}] Processing date {date.date()} ({i+1}/{total})", flush=True)
            dataset_builder.process_date(date, plot=args.plot, baseline=args.baseline, force=args.force)
    else:
        # OPT-2: Parallel path
        print(f"[OPT-2] Launching {args.workers} parallel workers for {total} dates.", flush=True)
        work_items = [
            (exp, date.strftime('%Y-%m-%d'), args.plot, args.baseline, args.force)
            for date in DATES
        ]
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_date_worker, item): item for item in work_items}
            for future in as_completed(futures):
                completed += 1
                date_str = futures[future][1]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[ERROR] {date_str}: {exc}", flush=True)
                if completed % 100 == 0 or completed == total:
                    elapsed = time.monotonic() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta_s = (total - completed) / rate if rate > 0 else float('inf')
                    print(
                        f"[{datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')}] "
                        f"Progress: {completed}/{total} — {rate:.1f} dates/s — ETA {eta_s/60:.1f} min",
                        flush=True
                    )
        elapsed = time.monotonic() - t0
        print(f"[OPT-2] Completed {total} dates in {elapsed/60:.1f} min ({total/elapsed:.1f} dates/s)", flush=True)
