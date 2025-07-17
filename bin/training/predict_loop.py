"""
Predict and save results for a full period by loading a trained model.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import glob
import xarray as xr
import pandas as pd
import torch
import argparse
import numpy as np
from torchvision.transforms import v2

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import (PREDICTION_DIR, 
                             RUNS_DIR, 
                             DATASET_BC_DIR, 
                             CONFIG,)
from iriscc.datautils import (remove_countries,
                              Data)

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
    args = parser.parse_args()


    run_dir = RUNS_DIR/f'{args.exp}/{args.test_name}/lightning_logs/version_best'
    checkpoint_dir = glob.glob(str(run_dir/f'checkpoints/best-checkpoint*.ckpt'))[0]

    device = 'cpu'
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
        test_name = f'{args.test_name}_{args.simu_test}'
        sample_dir = DATASET_BC_DIR / f'dataset_{args.exp}_test_{args.simu_test}' # bc or not
    else:
        test_name = args.test_name

    
    startdate = args.startdate
    enddate = args.enddate
    dates = pd.date_range(start=startdate, end=enddate, freq='D')
    if dates[-1] <= pd.Timestamp('2014-12-31'):
        period = 'historical'
    else:
        period = 'ssp585'

    if args.simu_test.startswith('gcm'):
        data_type = 'CNRM-CM6-1'
    elif args.simu_test.startswith('rcm'):
        data_type = 'ALADIN'
    
    ds, y = get_target_format(args.exp, dates=dates)
    y = np.expand_dims(y, axis= 0)
    
    for i, date in enumerate(dates):
        print(date)
        date_str = date.date().strftime('%Y%m%d')
        sample = glob.glob(str(sample_dir/f'sample_{date_str}.npz'))[0]
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

    ds.to_netcdf(PREDICTION_DIR/f'tas_day_{data_type}_{period}_r1i1p1f2_gr_{startdate}_{enddate}_{args.exp}_{test_name}.nc')
    