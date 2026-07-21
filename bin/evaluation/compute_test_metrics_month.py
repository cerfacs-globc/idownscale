"""
Evaluate input data x against target data y for raw, prediction and baseline data

date = 16/07/2025
author = Zoé GARCIA
"""

import sys
sys.path.append('.')

import os
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

from iriscc.inference import load_trained_module, predict_tensor
from iriscc.runtime_paths import (
    resolve_checkpoint_path,
    resolve_runtime_sample_dir,
    resolve_sample_file,
    resolve_statistics_dir,
)
from iriscc.transforms import DeMinMaxNormalisation, MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad, Log10Transform
from iriscc.settings import (CONFIG,
                             DATES_BC_TEST_HIST,
                             GRAPHS_DIR,
                             METRICS_DIR,
                             DATASET_DIR,
                             get_metrics_test_name,
                             )


def _seasonal_stack(values):
    if not values:
        return np.array([], dtype=np.float64)
    return np.stack(values)


def _seasonal_tensor_stack(values):
    if not values:
        return None
    return torch.stack(values)


def _corr_temporal_by_pixel(obs_tensor, pred_tensor, corr_metric):
    if obs_tensor is None or pred_tensor is None or obs_tensor.size(dim=0) < 2:
        return np.array([], dtype=np.float64)
    return np.stack([corr_metric(pred_tensor[:, j], obs_tensor[:, j]).cpu() for j in range(obs_tensor.size(dim=1))])


def _safe_mean(array_like):
    data = np.asarray(array_like, dtype=np.float64)
    if data.size == 0:
        return np.nan
    return float(np.nanmean(data))


def get_config(exp: str,
               test_name: str,
               simu_test: Optional[str],
               checkpoint_bundle: Optional[str] = None,
               device: str = 'cpu') -> Tuple[Optional[nn.Module], Optional[v2.Compose], str]:
    """
    Configure the model, transforms, and sample directory based on the experiment and test parameters.
    Args:
        exp (str): Experiment name.
        test_name (str): Test name (e.g., unet, baseline, gcm_raw).
        predict (bool): Whether to use a pretrained model.
        simu_test (Optional[str]): GCM test type (e.g., gcm or gcm_bc).
    Returns:
        Tuple[Optional[IRISCCLightningModule], Optional[v2.Compose], str]
    """
    model, transforms = None, None

    if test_name == 'era5_raw':
        sample_dir = DATASET_DIR / f'dataset_{exp}_30y'
    elif test_name.startswith('baseline') or test_name.endswith('_raw'):
        sample_dir = resolve_runtime_sample_dir(exp, test_name, simu_test=simu_test)
    else:
        checkpoint_dir = resolve_checkpoint_path(exp, test_name, checkpoint_bundle)
        model, hparams = load_trained_module(checkpoint_dir, device=device)

        statistics_dir = resolve_statistics_dir(hparams)
        transforms = v2.Compose([
            Log10Transform(hparams.get('channels', CONFIG[exp]['channels'])),
            MinMaxNormalisation(statistics_dir, hparams['output_norm']),
            LandSeaMask(hparams['mask'], hparams['fill_value']),
            FillMissingValue(hparams['fill_value']),
            Pad(hparams['fill_value'])
        ])

        sample_dir = resolve_runtime_sample_dir(exp, test_name, simu_test=simu_test, hparams=hparams)
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
        sample = resolve_sample_file(sample_dir, date_str)
        data = dict(np.load(sample, allow_pickle=True))
        x, y = data['x'], data['y']
        condition = np.isnan(y[0])

        if model: # unet, unet_gcm, unet_gcm_bc
            x, y = transforms((x, y))
            x = torch.unsqueeze(x, dim=0).float()
            y_hat = predict_tensor(model, x, model.hparams['hparams'], device).to(device)
            y_hat = y_hat.detach().cpu()
            unpad_func = UnPad(list(CONFIG[exp]['shape']))
            y, y_hat = unpad_func(y)[0].numpy(), unpad_func(y_hat[0])[0].numpy()
            hparams = model.hparams['hparams']
            if hparams.get('output_norm'):
                statistics_dir = resolve_statistics_dir(hparams)
                denorm = DeMinMaxNormalisation(statistics_dir, hparams['output_norm'])
                y = denorm((False, np.expand_dims(y, axis=0))).numpy()[0]
                y_hat = denorm((False, np.expand_dims(y_hat, axis=0))).numpy()[0]

        else: # baseline, era5_raw, gcm_raw
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
    parser.add_argument('--startdate', type=str, help='Start date (e.g., 20230101)', default=DATES_BC_TEST_HIST[0].strftime('%Y%m%d'))
    parser.add_argument('--enddate', type=str, help='End date (e.g., 20230101)', default=DATES_BC_TEST_HIST[-1].strftime('%Y%m%d'))
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')
    parser.add_argument('--test-name', type=str, help='Test name (e.g., unet, baseline, gcm_raw ...)')
    parser.add_argument('--simu-test', type=str, help='if predict (e.g., gcm or gcm_bc, rcm, rcm_bc)', default=None)
    parser.add_argument('--checkpoint-bundle', type=str, default=None, help='Optional portable checkpoint bundle directory.')
    args = parser.parse_args()

    exp = args.exp
    test_name = args.test_name
    simu_test = args.simu_test
    dates = pd.date_range(start=args.startdate, end=args.enddate, freq='D')

    transforms = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, transforms, sample_dir = get_config(exp, test_name, simu_test, args.checkpoint_bundle, device)

    test_name = get_metrics_test_name(test_name, simu_test)
    graph_dir = GRAPHS_DIR/f'metrics/{exp}/{test_name}/'
    metric_dir = METRICS_DIR/f'{exp}/mean_metrics'
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

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
    if dT:
        dT, dT_hat = np.stack(dT), np.stack(dT_hat)
        dT_summer = _seasonal_stack([dT[i-1,:] for i in i_summer if i > 0])
        dT_hat_summer = _seasonal_stack([dT_hat[i-1,:] for i in i_summer if i > 0])
        dT_winter = _seasonal_stack([dT[i-1,:] for i in i_winter if i > 0])
        dT_hat_winter = _seasonal_stack([dT_hat[i-1,:] for i in i_winter if i > 0])
        var = np.mean(np.abs(dT_hat), axis=0) - np.mean(np.abs(dT), axis=0)
        var_summer = np.full_like(var, np.nan) if dT_summer.size == 0 else np.mean(np.abs(dT_hat_summer), axis=0) - np.mean(np.abs(dT_summer), axis=0)
        var_winter = np.full_like(var, np.nan) if dT_winter.size == 0 else np.mean(np.abs(dT_hat_winter), axis=0) - np.mean(np.abs(dT_winter), axis=0)
    else:
        var = np.full(tuple(y_temporal[0].shape), np.nan, dtype=np.float64)
        var_summer = np.full_like(var, np.nan)
        var_winter = np.full_like(var, np.nan)

    y_temporal, y_hat_temporal = torch.stack(y_temporal), torch.stack(y_hat_temporal)
    corr_temporal = _corr_temporal_by_pixel(y_temporal, y_hat_temporal, corr)
    y_temporal_summer = _seasonal_tensor_stack([y_temporal[i,:] for i in i_summer])
    y_hat_temporal_summer = _seasonal_tensor_stack([y_hat_temporal[i,:] for i in i_summer])
    corr_temporal_summer = _corr_temporal_by_pixel(y_temporal_summer, y_hat_temporal_summer, corr)
    y_temporal_winter = _seasonal_tensor_stack([y_temporal[i,:] for i in i_winter])
    y_hat_temporal_winter = _seasonal_tensor_stack([y_hat_temporal[i,:] for i in i_winter])
    corr_temporal_winter = _corr_temporal_by_pixel(y_temporal_winter, y_hat_temporal_winter, corr)

    rmse_spatial = np.sqrt(rmse_spatial / len(dates))
    rmse_spatial_summer = np.full_like(rmse_spatial, np.nan) if len(i_summer) == 0 else np.sqrt(rmse_spatial_summer / len(i_summer))
    rmse_spatial_winter = np.full_like(rmse_spatial, np.nan) if len(i_winter) == 0 else np.sqrt(rmse_spatial_winter / len(i_winter))
    bias_spatial = bias_spatial / len(dates)
    bias_spatial_summer = np.full_like(bias_spatial, np.nan) if len(i_summer) == 0 else bias_spatial_summer / len(i_summer)
    bias_spatial_winter = np.full_like(bias_spatial, np.nan) if len(i_winter) == 0 else bias_spatial_winter / len(i_winter)

    rmse_temporal_summer = np.asarray([rmse_temporal[i] for i in i_summer], dtype=np.float64)
    rmse_temporal_winter = np.asarray([rmse_temporal[i] for i in i_winter], dtype=np.float64)
    corr_spatial_summer = np.asarray([corr_spatial[i] for i in i_summer], dtype=np.float64)
    corr_spatial_winter = np.asarray([corr_spatial[i] for i in i_winter], dtype=np.float64)

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
    d_mean = {'rmse_temporal_mean' : [_safe_mean(rmse_temporal), _safe_mean(rmse_temporal_summer), _safe_mean(rmse_temporal_winter)],
        'rmse_spatial_mean' : [np.nanmean(rmse_spatial),np.nanmean(rmse_spatial_summer), np.nanmean(rmse_spatial_winter)],
        'bias_spatial_mean' : [np.nanmean(bias_spatial),np.nanmean(bias_spatial_summer), np.nanmean(bias_spatial_winter)],
        'bias_spatial_std' : [np.nanstd(bias_spatial),np.nanstd(bias_spatial_summer), np.nanstd(bias_spatial_winter)],
        'corr_spatial_mean' : [_safe_mean(corr_spatial), _safe_mean(corr_spatial_summer), _safe_mean(corr_spatial_winter)],
        'corr_temporal_mean' : [_safe_mean(corr_temporal), _safe_mean(corr_temporal_summer), _safe_mean(corr_temporal_winter)],
        'variability_mean' : [_safe_mean(var), _safe_mean(var_summer), _safe_mean(var_winter)]}

    df = pd.DataFrame(d_mean, index = ['all', 'summer', 'winter'])
    df.to_csv(metric_dir/f'metrics_test_mean_monthly_{exp}_{test_name}.csv')
    print(df)
