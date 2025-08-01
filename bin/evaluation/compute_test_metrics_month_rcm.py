""" 
Evaluate input data x against high resolution rcm data y for rcm prediction data

date = 16/07/2025
author = Zoé GARCIA
"""

import sys
sys.path.append('.')

import os
import glob
import argparse
import xarray as xr
import torch
import numpy as np
import pandas as pd
from torchvision.transforms import v2
from torchmetrics import MeanSquaredError, PearsonCorrCoef

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, Log10Transform
from iriscc.settings import (CONFIG, 
                             GRAPHS_DIR, 
                             RUNS_DIR, 
                             METRICS_DIR, 
                             RCM_RAW_DIR)
from iriscc.transforms import UnPad
from iriscc.datautils import Data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for test period")
    parser.add_argument('--startdate', type=str, help='Start date (e.g., 20230101)', default='20000101')
    parser.add_argument('--enddate', type=str, help='End date (e.g., 20230101)', default='20141231')
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')   
    parser.add_argument('--test-name', type=str, help='Test name (e.g., unet, baseline, gcm_raw ...)')
    parser.add_argument('--simu-test', type=str, help='(e.g., gcm or gcm_bc)', default=None)
    args = parser.parse_args()

    exp = args.exp
    test_name = args.test_name
    simu_test = args.simu_test
    dates = pd.date_range(start=args.start_date, end=args.end_date, freq='D')
    get_data = Data(CONFIG[exp]['domain'])

    run_dir = RUNS_DIR/f'{exp}/{test_name}/lightning_logs/version_best'
    checkpoint_dir = glob.glob(str(run_dir/f'checkpoints/best-checkpoint*.ckpt'))[0]
    test_name = f'{test_name}_{simu_test}_pp'
    graph_dir = GRAPHS_DIR/f'metrics/{exp}/{test_name}/'
    metric_dir = METRICS_DIR/f'{exp}/mean_metrics'
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)


    model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
    model.eval()
    hparams = model.hparams['hparams']
    arch = hparams['model']
    domain = hparams['domain']
    if CONFIG[exp]['target'] == 'safran':
        domain = 'france_xy'
    transforms = v2.Compose([
                Log10Transform(hparams['channels']),
                MinMaxNormalisation(hparams['sample_dir'], hparams['output_norm']), 
                LandSeaMask(hparams['mask'], hparams['fill_value']),
                FillMissingValue(hparams['fill_value']),
                Pad(hparams['fill_value'])
                ])
    device = 'cpu'
    sample_dir = hparams['sample_dir']


    rmse = MeanSquaredError(squared=False).to(device)
    corr = PearsonCorrCoef().to(device)

    rmse_temporal = []  
    rmse_spatial = []  
    rmse_spatial_summer = []
    rmse_spatial_winter = []
    bias_spatial = []
    bias_spatial_summer = []
    bias_spatial_winter = []
    corr_spatial = []
    corr_spatial_summer = []
    corr_spatial_winter = []
    y_temporal = []
    y_hat_temporal = []
    i_summer = []
    i_winter = []

    dates_month = dates.to_period('M').astype(str).unique()
  
    df_dates = pd.DataFrame({'date': dates})
    df_dates['year'] = df_dates['date'].dt.year
    df_dates['month'] = df_dates['date'].dt.month
    df_dates['day'] = df_dates['date'].dt.day

    for i, ((year, month), group) in enumerate(df_dates.groupby(['year', 'month'])):
        if month in [6,7,8]:
            i_summer.append(i)
        if month in [1,2,12]:
            i_winter.append(i)

        daily_y = []
        daily_y_hat = []
        for day in group['day']:
            date_str = f'{year}{month:02d}{day:02d}'
            print(date_str)
            sample = glob.glob(str(sample_dir/f'sample_{date_str}.npz'))[0]
            data = dict(np.load(sample), allow_pickle=True)
            x = data['x']
            y = []
            for var in CONFIG[exp]['target_vars']:
                file = glob.glob(str(RCM_RAW_DIR/f'ALADIN_reformat/{var}*nc'))[0]
                ds = xr.open_dataset(file)
                ds = ds.sel(time=ds.time.dt.date == pd.Timestamp(date_str).date())
                ds = ds.isel(time=0)
                y.append(ds[var].values)

            y = np.stack(y, axis=0)
            condition = np.isnan(y[0])

            x, y = transforms((x, y))
            x = torch.unsqueeze(x, dim=0).float()
            y_hat = model(x.to(device)).to(device)
            y_hat = y_hat.detach().cpu()
            unpad_func = UnPad(list(CONFIG[exp]['shape']))
            y, y_hat = unpad_func(y)[0].numpy(), unpad_func(y_hat[0])[0].numpy()
                
            y[condition] = np.nan
            y_hat[condition] = np.nan
            daily_y.append(y)
            daily_y_hat.append(y_hat)

        y = np.mean(np.stack(daily_y), axis=0)
        y_hat = np.mean(np.stack(daily_y_hat), axis=0)

        h, w = y.shape

        # compute metrics
        ## spatial metrics
        error = (y_hat - y)
        error_squared = error ** 2
        if len(rmse_spatial) == 0:
            rmse_spatial = error_squared
            bias_spatial = error
        else:
            rmse_spatial += error_squared
            bias_spatial += error
        if month in [6,7,8]:
            if len(rmse_spatial_summer) == 0:
                rmse_spatial_summer = error_squared
                bias_spatial_summer = error
            else:
                rmse_spatial_summer += error_squared
                bias_spatial_summer += error
        elif month in [1,2,12]:
            if len(rmse_spatial_winter) == 0:
                rmse_spatial_winter = error_squared
                bias_spatial_winter = error
            else:
                rmse_spatial_winter += error_squared
                bias_spatial_winter += error

        ## temporal metrics
        y, y_hat = torch.tensor(y), torch.tensor(y_hat)
        y_flat, y_hat_flat = y[~torch.isnan(y)].to(device), y_hat[~torch.isnan(y_hat)].to(device)
        rmse_value = rmse(y_hat_flat, y_flat).item()
        corr_value = corr(y_hat_flat - torch.mean(y_hat_flat), y_flat - torch.mean(y_flat)).item()
        rmse_temporal.append(rmse_value)
        corr_spatial.append(corr_value)

        y_temporal.append(y_flat)
        y_hat_temporal.append(y_hat_flat)


    dT = [y_temporal[i] - y_temporal[i-1] for i in range(len(y_temporal)-1)]
    dT_hat = [y_hat_temporal[i] - y_hat_temporal[i-1] for i in range(len(y_hat_temporal)-1)]
    dT, dT_hat = np.stack(dT), np.stack(dT_hat)
    dT_summer = np.stack([dT[i-1,:] for i in i_summer])
    dT_hat_summer = np.stack([dT_hat[i-1,:] for i in i_summer])
    dT_winter = np.stack([dT[i-1,:] for i in i_winter])
    dT_hat_winter = np.stack([dT_hat[i-1,:] for i in i_winter])
    var = np.mean(dT_hat, axis=0) - np.mean(dT, axis=0)
    var_summer = np.mean(dT_hat_summer, axis=0) - np.mean(dT_summer, axis=0)
    var_winter = np.mean(dT_hat_winter, axis=0) - np.mean(dT_winter, axis=0)

    y_temporal, y_hat_temporal = torch.stack(y_temporal), torch.stack(y_hat_temporal)
    corr_temporal = [corr(y_hat_temporal[:,j], y_temporal[:,j]).cpu() for j in range(y_temporal.size(dim=1))]
    corr_temporal = np.stack(corr_temporal)
    y_temporal_summer = torch.stack([y_temporal[i,:] for i in i_summer])
    y_hat_temporal_summer = torch.stack([y_hat_temporal[i,:] for i in i_summer])
    corr_temporal_summer = [corr(y_hat_temporal_summer[:,j], y_temporal_summer[:,j]).cpu() for j in range(y_temporal.size(dim=1))]
    corr_temporal = np.stack(corr_temporal_summer)
    y_temporal_winter = torch.stack([y_temporal[i,:] for i in i_winter])
    y_hat_temporal_winter = torch.stack([y_hat_temporal[i,:] for i in i_winter])
    corr_temporal_winter = [corr(y_hat_temporal_winter[:,j], y_temporal_winter[:,j]).cpu() for j in range(y_temporal.size(dim=1))]
    corr_temporal = np.stack(corr_temporal_winter)

    rmse_spatial = np.sqrt(rmse_spatial / len(dates))
    rmse_spatial_summer = np.sqrt(rmse_spatial_summer / len(i_summer))
    rmse_spatial_winter = np.sqrt(rmse_spatial_winter / len(i_winter))
    bias_spatial = bias_spatial / len(dates)
    bias_spatial_summer = bias_spatial_summer / len(i_summer)
    bias_spatial_winter = bias_spatial_winter / len(i_winter)

    rmse_temporal_summer = np.stack([rmse_temporal[i] for i in i_summer])
    rmse_temporal_winter = np.stack([rmse_temporal[i] for i in i_winter])
    corr_spatial_summer = np.stack([corr_spatial[i] for i in i_summer])
    corr_spatial_winter = np.stack([corr_spatial[i] for i in i_winter])

    # Save temporal and spatial values only for all period
    d = {'rmse_temporal': rmse_temporal,
            'rmse_spatial': rmse_spatial.flatten(),
            'bias_spatial': bias_spatial.flatten(),
            'corr_temporal': corr_temporal,
            'corr_spatial': corr_spatial,
            'variability': var,
            'dates' : dates_month}
    np.savez(metric_dir/f'metrics_test_monthly_{exp}_{test_name}.npz', **d)

    # Save mean values
    d_mean = {'rmse_temporal_mean' : [np.mean(rmse_temporal), np.mean(rmse_temporal_summer), np.mean(rmse_temporal_winter)],
        'rmse_spatial_mean' : [np.nanmean(rmse_spatial),np.nanmean(rmse_spatial_summer), np.nanmean(rmse_spatial_winter)],
        'bias_spatial_mean' : [np.nanmean(bias_spatial),np.nanmean(bias_spatial_summer), np.nanmean(bias_spatial_winter)],
        'bias_spatial_std' : [np.nanstd(bias_spatial),np.nanstd(bias_spatial_summer), np.nanstd(bias_spatial_winter)],
        'corr_spatial_mean' : [np.mean(corr_spatial), np.mean(corr_spatial_summer), np.mean(corr_spatial_winter)],
        'corr_temporal_mean' : [np.mean(corr_temporal), np.mean(corr_temporal_summer), np.mean(corr_temporal_winter)],
        'variability_mean' : [np.mean(var), np.mean(var_summer), np.mean(var_winter)]}

    df = pd.DataFrame(d_mean, index = ['all', 'summer', 'winter'])
    df.to_csv(metric_dir/f'metrics_test_mean_monthly_{exp}_{test_name}.csv')
    print(df)
