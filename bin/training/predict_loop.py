import sys
sys.path.append('.')

import glob
import xarray as xr
import torch
import argparse
import numpy as np
from torchvision.transforms import v2

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import (PREDICTION_DIR, 
                             TARGET_SAFRAN_FILE, 
                             TARGET_SIZE, 
                             RUNS_DIR, 
                             DATASET_BC_DIR, 
                             DATES_BC_TEST_HIST,
                             DATES_BC_TRAIN_HIST)
from iriscc.datautils import standardize_longitudes, remove_countries

parser = argparse.ArgumentParser(description="Predict and plot results for full period")
parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')   
parser.add_argument('--test-name', type=str, help='Test name (e.g., mask_continents)')
parser.add_argument('--cmip6-test', type=str, help='cmip6 or cmip6_bc', default=None)
args = parser.parse_args()

dates = DATES_BC_TRAIN_HIST

run_dir = RUNS_DIR/f'{args.exp}/{args.test_name}/lightning_logs/version_best'
checkpoint_dir = glob.glob(str(run_dir/f'checkpoints/best-checkpoint*.ckpt'))[0]

model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
model.eval()
hparams = model.hparams['hparams']
arch = hparams['model']

transforms = v2.Compose([
            MinMaxNormalisation(hparams['sample_dir'], hparams['output_norm']), 
            LandSeaMask(hparams['mask'], hparams['fill_value']),
            FillMissingValue(hparams['fill_value']),
            Pad(hparams['fill_value'])
            ])

sample_dir = hparams['sample_dir']
if args.cmip6_test == 'cmip6' or args.cmip6_test == 'cmip6_bc':
    test_name = f'{args.test_name}_{args.cmip6_test}'
    sample_dir = DATASET_BC_DIR / f'dataset_{args.exp}_test_{args.cmip6_test}' # bc or not
else:
    test_name = args.test_name
device = 'cpu'

startdate = dates[0].date().strftime('%d/%m/%Y')
enddate = dates[-1].date().strftime('%d/%m/%Y')
period = f'{startdate} - {enddate}'

ds_target = xr.open_dataset(TARGET_SAFRAN_FILE).isel(time=0)
ds_target = standardize_longitudes(ds_target)
y = ds_target.tas.values

ds = xr.Dataset(
    data_vars={'tas': (['time', 'y', 'x'], np.empty((len(dates), y.shape[0], y.shape[1])))},
    coords={"time" : dates,
                "y" : ds_target.y.values,
                "x" : ds_target.x.values})
y = remove_countries(y)
condition = np.isnan(y)
y = np.expand_dims(y, axis= 0)

for i, date in enumerate(dates):
    print(date)
    date_str = date.date().strftime('%Y%m%d')
    sample = glob.glob(str(sample_dir/f'sample_{date_str}.npz'))[0]
    data = dict(np.load(sample), allow_pickle=True)

    x = data['x']
    x, _ = transforms((x, y))
    x = torch.unsqueeze(x, dim=0).float()
    y_hat = model(x.to(device)).to(device)
    y_hat = y_hat.detach().cpu()

    unpad_func = UnPad(TARGET_SIZE)
    y_hat = unpad_func(y_hat[0])[0].numpy()
    y_hat[condition] = np.nan

    ds.tas[i] = y_hat

ds_ref = xr.open_dataset(PREDICTION_DIR/f'tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp3_swinunet_all_cmip6_bc.nc')
ds_all = xr.concat([ds, ds_ref], dim='time')

ds_all.to_netcdf(PREDICTION_DIR/f'tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101_20141231_{args.exp}_{test_name}.nc')