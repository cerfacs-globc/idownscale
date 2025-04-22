import sys
sys.path.append('.')

import os
import glob
import torch
import numpy as np
import pandas as pd
from torchmetrics import MeanSquaredError, PearsonCorrCoef

from iriscc.transforms import DomainCrop
from iriscc.settings import (DATES_TEST, 
                             DATES_BC_TEST_HIST,
                             CONFIG, 
                             GRAPHS_DIR, 
                             TARGET_SIZE, 
                             RUNS_DIR, 
                             METRICS_DIR, 
                             DATASET_BC_DIR,
                             DATASET_EXP3_30Y_DIR,
                             DATASET_EXP4_30Y_DIR)


exp = str(sys.argv[1]) # ex : exp 1
input = str(sys.argv[2]) # cmip6 raw, era5 raw
target = str(sys.argv[3])
crop = str(sys.argv[4])

if input == 'cmip6_raw':
    sample_dir = DATASET_BC_DIR / f'dataset_{exp}_test_cmip6'
elif input == 'era5_raw':
    if exp == 'exp3':
        sample_dir = DATASET_EXP3_30Y_DIR
    elif exp == 'exp4':
        sample_dir = DATASET_EXP4_30Y_DIR

domain = CONFIG[target]['domain'][crop]

if target == 'eobs':
    domaincrop = DomainCrop(sample_dir, domain)


graph_dir = GRAPHS_DIR/f'metrics/{exp}/{input}/'
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

dates = DATES_BC_TEST_HIST
startdate = dates[0].date().strftime('%d/%m/%Y')
enddate = dates[-1].date().strftime('%d/%m/%Y')
period = f'{startdate} - {enddate}'

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
        
    daily_y = []
    daily_y_hat = []
    for day in group['day']:
        date_str = f'{year}{month:02d}{day:02d}'
        print(date_str)
        sample = glob.glob(str(sample_dir/f'sample_{date_str}.npz'))[0]
        data = dict(np.load(sample), allow_pickle=True)
        x, y = data['x'], data['y']
        y_hat = x[1] 
        condition = np.isnan(y[0])
        y_hat[condition] = np.nan


        y = np.expand_dims(y, axis = 0)
        y_hat = np.expand_dims(y_hat, axis = 0)
        if target == 'eobs':
            y_hat, y = domaincrop((y_hat, y))
        y_hat, y = y_hat[0], y[0]

        daily_y.append(y)
        daily_y_hat.append(y_hat)

    y = np.mean(np.stack(daily_y), axis=0)
    y_hat = np.mean(np.stack(daily_y_hat), axis=0)

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

print(bias_spatial[~np.isnan(bias_spatial)].flatten())
# Save temporal and spatial values only for all period
d = {'rmse_temporal': rmse_temporal,
    'bias_spatial': bias_spatial[~np.isnan(bias_spatial)].flatten(),
    'corr_temporal': corr_temporal,
    'corr_spatial': corr_spatial,
    'variability': var}
np.savez(metric_dir/f'metrics_test_monthly_{exp}_{input}.npz', **d)

# Save mean values
d_mean = {'rmse_temporal_mean' : [np.mean(rmse_temporal), np.mean(rmse_temporal_summer), np.mean(rmse_temporal_winter)],
    'rmse_spatial_mean' : [np.nanmean(rmse_spatial),np.nanmean(rmse_spatial_summer), np.nanmean(rmse_spatial_winter)],
    'bias_spatial_mean' : [np.nanmean(bias_spatial),np.nanmean(bias_spatial_summer), np.nanmean(bias_spatial_winter)],
    'bias_spatial_std' : [np.nanstd(bias_spatial),np.nanstd(bias_spatial_summer), np.nanstd(bias_spatial_winter)],
    'corr_spatial_mean' : [np.mean(corr_spatial), np.mean(corr_spatial_summer), np.mean(corr_spatial_winter)],
    'corr_temporal_mean' : [np.mean(corr_temporal), np.mean(corr_temporal_summer), np.mean(corr_temporal_winter)],
    'variability_mean' : [np.mean(var), np.mean(var_summer), np.mean(var_winter)]}

df = pd.DataFrame(d_mean, index = ['all', 'summer', 'winter'])
df.to_csv(metric_dir/f'metrics_test_mean_monthly_{exp}_{input}.csv')
print(df)

