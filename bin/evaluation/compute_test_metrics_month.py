""" 
Evaluate input data x against target data y for raw, prediction and baseline data
"""

import sys
sys.path.append('.')

import os
import glob
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Optional
from torch import nn
from torchvision.transforms import Compose
import pandas as pd
from typing import Optional
from torchvision.transforms import v2
from torchmetrics import MeanSquaredError, PearsonCorrCoef

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, DomainCrop, UnPad
from iriscc.settings import (CONFIG, 
                             GRAPHS_DIR, 
                             RUNS_DIR, 
                             METRICS_DIR, 
                             DATASET_BC_DIR,
                             DATASET_DIR)


def get_config(exp: str, test_name: str, cmip6_test: Optional[str]) -> Tuple[Optional[IRISCCLightningModule], Optional[v2.Compose], str]:
    """
    Configure the model, transforms, and sample directory based on the experiment and test parameters.
    Args:
        exp (str): Experiment name.
        test_name (str): Test name (e.g., unet, baseline, cmip6_raw).
        predict (bool): Whether to use a pretrained model.
        cmip6_test (Optional[str]): CMIP6 test type (e.g., cmip6 or cmip6_bc).
    Returns:
        Tuple[Optional[IRISCCLightningModule], Optional[v2.Compose], str]
    """
    model, transforms = None, None

    if test_name.startswith('baseline'):
        sample_dir = DATASET_DIR / f'dataset_{exp}_baseline'
    elif test_name == 'cmip6_raw':
        sample_dir = DATASET_BC_DIR / f'dataset_{exp}_test_cmip6'
    elif test_name == 'era5_raw':
        sample_dir = DATASET_DIR / f'dataset_{exp}_30y'
    else:
        run_dir = RUNS_DIR / f'{exp}/{test_name}/lightning_logs/version_best'
        checkpoint_dir = glob.glob(str(run_dir / 'checkpoints/best-checkpoint*.ckpt'))[0]
        model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
        model.eval()
        hparams = model.hparams['hparams']
        
        transforms = v2.Compose([
            MinMaxNormalisation(hparams['sample_dir'], hparams['output_norm']), 
            LandSeaMask(hparams['mask'], hparams['fill_value']),
            FillMissingValue(hparams['fill_value']),
            DomainCrop(hparams['sample_dir'], hparams['domain_crop']),
            Pad(hparams['fill_value'])
        ])
        
        if cmip6_test:
            sample_dir = DATASET_BC_DIR / f'dataset_{exp}_test_{cmip6_test}'  # bc or not
        else:
            sample_dir = hparams['sample_dir']
    return model, transforms, sample_dir

def preprocess(year:int,
                month:int,
                sample_dir: Path,
                model: Optional[nn.Module],
                transforms: Optional[Compose]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses input data and generates predictions using a given model.
    Args:
        date (datetime.date): The date corresponding to the sample to process.
        sample_dir (Path): Directory containing the sample `.npz` files.
        model (Optional[nn.Module]): PyTorch model used for predictions or None.
        transforms (Optional[Compose]): Transformations to apply to the input data (x, y).
    Returns:
        Tuple[np.ndarray, np.ndarray]
    """
    daily_y = []
    daily_y_hat = []
    for day in group['day']:
        date_str = f'{year}{month:02d}{day:02d}'
        print(date_str)
        sample = glob.glob(str(sample_dir/f'sample_{date_str}.npz'))[0]
        data = dict(np.load(sample), allow_pickle=True)
        x, y = data['x'], data['y']
        condition = np.isnan(y[0])

        if model: # unet, unet_cmip6, unet_cmip6_bc
            x, y = transforms((x, y))
            x = torch.unsqueeze(x, dim=0).float()
            y_hat = model(x.to(device)).to(device)
            y_hat = y_hat.detach().cpu()

            if exp == 'exp3':
                unpad_func = UnPad(list(CONFIG['safran']['shape']['france_xy']))
                y, y_hat = unpad_func(y)[0].numpy(), unpad_func(y_hat[0])[0].numpy()
            else:
                y, y_hat = y[0,...].numpy(), y_hat[0,0,...].numpy()

        else: # baseline, era5_raw, cmip6_raw
            y = y[0]
            y_hat = x[-1] # all .npz datasets are {'x': x, 'y': y}-like
            
        y[condition] = np.nan
        y_hat[condition] = np.nan
        daily_y.append(y)
        daily_y_hat.append(y_hat)

    y = np.mean(np.stack(daily_y), axis=0)
    y_hat = np.mean(np.stack(daily_y_hat), axis=0)

    return y, y_hat


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Compute metrics for test period")
    parser.add_argument('--start-date', type=str, help='Start date (e.g., 2023-01-01)', default='2000-01-01')
    parser.add_argument('--end-date', type=str, help='End date (e.g., 2023-01-01)', default='2014-12-31')
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')   
    parser.add_argument('--test-name', type=str, help='Test name (e.g., unet, baseline, cmip6_raw ...)')
    parser.add_argument('--cmip6-test', type=str, help='if predict (e.g., cmip6 or cmip6_bc)', default=None)
    args = parser.parse_args()

    exp = args.exp
    test_name = args.test_name
    cmip6_test = args.cmip6_test
    dates = pd.date_range(start=args.start_date, end=args.end_date, freq='D')

    transforms = None
    model, transforms, sample_dir = get_config(exp, test_name, cmip6_test)

    if cmip6_test:
        test_name = f'{test_name}_{cmip6_test}'
    graph_dir = GRAPHS_DIR/f'metrics/{exp}/{test_name}/'
    metric_dir = METRICS_DIR/f'{exp}/mean_metrics'
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    device = 'cpu'
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

    df_dates = pd.DataFrame({'date': dates})
    df_dates['year'] = df_dates['date'].dt.year
    df_dates['month'] = df_dates['date'].dt.month
    df_dates['day'] = df_dates['date'].dt.day

    #boucle sur le duo mois années 
    ## boucle interne sur les jours du mois en faisant la moyenne

    for i, ((year, month), group) in enumerate(df_dates.groupby(['year', 'month'])):
        if month in [6,7,8]:
            i_summer.append(i)
        if month in [1,2,12]:
            i_winter.append(i)

        y, y_hat = preprocess(year, month, sample_dir, model, transforms)

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
    var = np.mean(np.abs(dT_hat), axis=0) - np.mean(np.abs(dT), axis=0)
    var_summer = np.mean(np.abs(dT_hat_summer), axis=0) - np.mean(np.abs(dT_summer), axis=0)
    var_winter = np.mean(np.abs(dT_hat_winter), axis=0) - np.mean(np.abs(dT_winter), axis=0)

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
        'variability': var}
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

