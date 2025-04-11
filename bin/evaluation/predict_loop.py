import sys
sys.path.append('.')

import glob
import xarray as xr
import torch
import numpy as np
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import PREDICTION_DIR, TARGET_SAFRAN_FILE, TARGET_SIZE, RUNS_DIR, DATASET_EXP3_30Y_DIR, DATASET_EXP1_30Y_DIR, CONFIG, DATASET_BC_DIR, DATES_BC_TEST_HIST, DATES_BC_TEST_FUTURE
from iriscc.plotutils import plot_map_contour
from iriscc.datautils import standardize_longitudes, remove_countries

    
exp = str(sys.argv[1]) # ex : exp 1
test_name = str(sys.argv[2]) # ex : mask_continents
cmip6_test = str(sys.argv[3]) # no, cmip6, cmip6_bc

dates = DATES_BC_TEST_FUTURE

run_dir = RUNS_DIR/f'{exp}/{test_name}/lightning_logs/version_best'
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
if cmip6_test == 'cmip6' or cmip6_test == 'cmip6_bc':
    test_name = f'{test_name}_{cmip6_test}'
    sample_dir = DATASET_BC_DIR / f'dataset_{exp}_test_{cmip6_test}' # bc or not
    pp = f'_{cmip6_test}'
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

ds.to_netcdf(PREDICTION_DIR/f'tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101_21001231_{test_name}.nc')