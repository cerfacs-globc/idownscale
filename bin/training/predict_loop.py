"""
Predict and save results for a full period by loading a trained model.

date : 16/07/2025
author : Zoé GARCIA
"""

import argparse
import datetime
import sys

sys.path.append('.')

import numpy as np
import pandas as pd
import pathlib
import torch
import xarray as xr
from torchvision.transforms import v2

# REQUIRED FIX for PyTorch 2.6+ to allow loading checkpoints with Path objects
torch.serialization.add_safe_globals([pathlib.PosixPath])

from iriscc.datautils import Data, remove_countries
from iriscc.lightning_module import IRISCCLightningModule
from iriscc.settings import (CONFIG, DATASET_BC_DIR, PREDICTION_DIR,
                             RUNS_DIR)
from iriscc.transforms import (FillMissingValue, LandSeaMask, MinMaxNormalisation,
                               Pad, UnPad)

def get_target_format(exp:str, dates):
    get_data = Data(CONFIG[exp]['domain'])
    ds_target = get_data.get_target_dataset(target = CONFIG[exp]['target'], 
                                            var = CONFIG[exp]['target_vars'][0],
                                            date=pd.Timestamp('2014-12-31 00:00:00'))
    y = ds_target.tas.values
    
    if 'x' in ds_target.dims:
        ds = xr.Dataset(
            data_vars={'tas': (['time', 'y', 'x'], np.empty((len(dates), y.shape[0], y.shape[1])))},
            coords={"time" : dates,
                        "y" : ds_target.y.values,
                        "x" : ds_target.x.values})
        if exp == 'exp3':
            y = remove_countries(y)
    elif 'lon' in ds_target.dims:
        ds = xr.Dataset(
            data_vars={'tas': (['time', 'lat', 'lon'], np.empty((len(dates), y.shape[0], y.shape[1])))},
            coords={"time" : dates,
                        "lat" : ds_target.lat.values,
                        "lon" : ds_target.lon.values})
    return ds, y


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Predict and plot results for full period")
    parser.add_argument('--startdate', type=str, help='Start date (e.g., 20230101)', default='20000101')
    parser.add_argument('--enddate', type=str, help='End date (e.g., 20230101)', default='20141231')
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')
    parser.add_argument('--test-name', type=str, help='Test name (e.g., mask_continents)')
    parser.add_argument('--simu-test', type=str, help='gcm or gcm_bc, rcm, rcm_bc', default=None)
    parser.add_argument('--force', action='store_true', help='Force prediction regeneration')
    args = parser.parse_args()

    # Define output filename first to check for existence
    startdate = args.startdate
    enddate = args.enddate
    if pd.to_datetime(enddate) <= pd.Timestamp('2014-12-31'):
        period = 'historical'
    else:
        period = 'ssp585'

    if args.simu_test.startswith('gcm'):
        data_type = 'CNRM-CM6-1'
    elif args.simu_test.startswith('rcm'):
        data_type = 'ALADIN'
    
    test_name = args.test_name
    if args.simu_test is not None:
        test_name = f'{args.test_name}_{args.simu_test}'

    output_file = PREDICTION_DIR/f'tas_day_{data_type}_{period}_r1i1p1f2_gr_{startdate}_{enddate}_{args.exp}_{test_name}.nc'
    
    if output_file.exists() and not args.force:
        print(f"Skipping INFERENCE: {output_file} already exists. Use --force to overwrite.", flush=True)
        sys.exit(0)

    run_dir = RUNS_DIR/f'{args.exp}/{args.test_name}/lightning_logs/version_best'
    checkpoint_dir = next(run_dir.glob('checkpoints/best-checkpoint*.ckpt'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)
    model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir, map_location=device)
    model.eval()
    hparams = model.hparams['hparams']

    transforms = v2.Compose([
                MinMaxNormalisation(hparams['sample_dir'], hparams['output_norm']), 
                LandSeaMask(hparams['mask'], hparams['fill_value']),
                FillMissingValue(hparams['fill_value']),
                Pad(hparams['fill_value'])
                ])

    sample_dir = hparams['sample_dir']
    if args.simu_test is not None:
        sample_dir = DATASET_BC_DIR / f'dataset_{args.exp}_test_{args.simu_test}' # bc or not

    dates = pd.date_range(start=startdate, end=enddate, freq='D')
    
    ds, y = get_target_format(args.exp, dates=dates)
    y = np.expand_dims(y, axis= 0)
    
    # Ensure output directory exists
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    
    total = len(dates)
    for i, date in enumerate(dates):
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')
        print(f"[{now_str}] [INFERENCE] Processing {date.date()} ({i+1}/{total})", flush=True)
        date_str = date.date().strftime('%Y%m%d')
        sample = next(sample_dir.glob(f'sample_{date_str}.npz'))
        data = dict(np.load(sample), allow_pickle=True)

        x = data['x']

        x, y = transforms((x, y))
        condition = y[0] == 0
        x = torch.unsqueeze(x, dim=0).float()
        y_hat = model(x.to(device)).to(device)
        y_hat = y_hat.detach().cpu()

        unpad_func = UnPad(list(CONFIG[args.exp]['shape']))
        y_hat = unpad_func(y_hat[0])[0].numpy()
        y_hat[condition] = np.nan
        ds.tas[i] = y_hat

    ds.to_netcdf(output_file)
    