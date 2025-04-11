import sys
sys.path.append('.')

import os
import glob
import torch
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchmetrics import MeanSquaredError, PearsonCorrCoef

from iriscc.settings import (DATES_TEST, 
                             GRAPHS_DIR, 
                             METRICS_DIR, 
                             DATASET_EXP3_BASELINE_DIR,
                             CONFIG,
                             DATASET_EXP4_BASELINE_DIR,
                             CONFIG)
from iriscc.transforms import DomainCrop
from iriscc.plotutils import plot_map_contour


exp = str(sys.argv[1]) # ex : exp 1
target = str(sys.argv[2])
crop = str(sys.argv[3])

sample_dir = DATASET_EXP4_BASELINE_DIR
domain = CONFIG[target]['domain'][crop]

if target == 'eobs':
    test_name = f'baseline_{crop}'
    domaincrop = DomainCrop(sample_dir, domain)

elif target == 'safran': 
    test_name = 'baseline'

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

dates = DATES_TEST
startdate = dates[0].date().strftime('%d/%m/%Y')
enddate = dates[-1].date().strftime('%d/%m/%Y')
period = f'{startdate} - {enddate}'

for i, date in enumerate(dates):
    print(date)
    if date.month in [6,7,8]:
        i_summer.append(i)
    if date.month in [1,2,12]:
        i_winter.append(i)
    date_str = date.date().strftime('%Y%m%d')
    sample = glob.glob(str(sample_dir/f'sample_{date_str}.npz'))[0]
    data = dict(np.load(sample), allow_pickle=True)
    y_hat, y = data['y_hat'], data['y']
    condition = np.isnan(y)
    y_hat[condition] = np.nan

    y = np.expand_dims(y, axis = 0)
    y_hat = np.expand_dims(y_hat, axis = 0)
    if target == 'eobs':
        y_hat, y = domaincrop((y_hat, y))
    y_hat, y = y_hat[0], y[0]

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
    if date.month in [6,7,8]:
        if len(rmse_spatial_summer) == 0:
            rmse_spatial_summer = error_squared
            bias_spatial_summer = error
        else:
            rmse_spatial_summer += error_squared
            bias_spatial_summer += error
    elif date.month in [1,2,12]:
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

    print(y_flat.shape , y_hat_flat.shape)
    y_temporal.append(y_flat)
    y_hat_temporal.append(y_hat_flat)


y_temporal, y_hat_temporal = torch.stack(y_temporal), torch.stack(y_hat_temporal)
corr_temporal = [corr(y_hat_temporal[i,:], y_temporal[i,:]).cpu() for i in range(y_temporal.size(dim=0))]
corr_temporal = np.stack(corr_temporal)
corr_temporal_summer = [corr(y_hat_temporal[i,:], y_temporal[i,:]).cpu() for i in i_summer]
corr_temporal = np.stack(corr_temporal_summer)
corr_temporal_winter = [corr(y_hat_temporal[i,:], y_temporal[i,:]).cpu() for i in i_winter]
corr_temporal = np.stack(corr_temporal_winter)

rmse_spatial = np.sqrt(rmse_spatial / len(dates))
rmse_spatial_summer = np.sqrt(rmse_spatial_summer / len(i_summer))
rmse_spatial_winter = np.sqrt(rmse_spatial_winter / len(i_winter))
bias_spatial = bias_spatial / len(dates)
bias_spatial_summer = bias_spatial_summer / len(i_summer)
bias_spatial_winter = bias_spatial_winter / len(i_winter)

rmse_temporal_summer = [rmse_temporal[i] for i in i_summer]
rmse_temporal_winter = [rmse_temporal[i] for i in i_winter]
corr_spatial_summer = [corr_spatial[i] for i in i_summer]
corr_spatial_winter = [corr_spatial[i] for i in i_winter]

# Save temporal and spatial values only for all period
d = {'rmse_temporal': rmse_temporal,
    'bias_spatial': bias_spatial[~np.isnan(bias_spatial)].flatten(),
    'corr_temporal': corr_temporal,
    'corr_spatial': corr_spatial
    }
np.savez(metric_dir/f'metrics_test_daily_{exp}_{test_name}.npz', **d)

# Save mean values
d_mean = {'rmse_temporal_mean' : [np.mean(rmse_temporal), np.mean(rmse_temporal_summer), np.mean(rmse_temporal_winter)],
    'rmse_spatial_mean' : [np.nanmean(rmse_spatial),np.nanmean(rmse_spatial_summer), np.nanmean(rmse_spatial_winter)],
    'bias_spatial_mean' : [np.nanmean(bias_spatial),np.nanmean(bias_spatial_summer), np.nanmean(bias_spatial_winter)],
    'bias_spatial_std' : [np.nanstd(bias_spatial),np.nanstd(bias_spatial_summer), np.nanstd(bias_spatial_winter)],
    'corr_spatial_mean' : [np.mean(corr_spatial), np.mean(corr_spatial_summer), np.mean(corr_spatial_winter)],
    'corr_temporal_mean' : [np.mean(corr_temporal), np.mean(corr_temporal_summer), np.mean(corr_temporal_winter)]}

df = pd.DataFrame(d_mean, index = ['all', 'summer', 'winter'])
df.to_csv(metric_dir/f'metrics_test_mean_daily_{exp}_{test_name}.csv')
print(df)

# Spatial distribution
## RMSE
levels = np.arange(0, 9.75, 0.25) 
colors = [
    '#a1d99b', '#41ab5d', '#006d2c',  # Vert clair -> foncé
    '#ffeda0', '#feb24c', '#d45f00',
    '#fc9272', '#de2d26', '#a50f15',   # Rouge clair -> foncé
    '#9ecae1', '#3182bd', '#08519c'
]
fig, ax = plot_map_contour(rmse_spatial,
                    domain = domain,
                    data_projection = CONFIG[target]['data_projection'],
                    fig_projection = CONFIG[target]['fig_projection'][crop],
                    title = f'baseline ({target} Evalutaion)',
                    cmap=mcolors.ListedColormap(colors[:len(levels) - 1]),
                    levels=levels ,
                    var_desc='Bias (K)')
ax.text(0.03, 0.07, f"Mean spatial RMSE: {np.nanmean(rmse_spatial):.2f}", 
        transform=ax.transAxes, fontsize=10, verticalalignment='top', zorder=10, 
        horizontalalignment='left', color = 'red', 
        bbox={'facecolor': 'white', 'pad': 5, 'edgecolor' : 'white'})
plt.savefig(f"{graph_dir}/daily_spatial_rmse_distribution.png")

## Bias
fig, ax = plot_map_contour(bias_spatial,
                    domain = domain,
                    data_projection = CONFIG[target]['data_projection'],
                    fig_projection = CONFIG[target]['fig_projection'][crop],
                    title = f'baseline ({target} Evalutaion)',
                    cmap='BrBG',
                    levels= np.linspace(-10,10,11),
                    var_desc='Bias (K)')
ax.text(0.03, 0.07, f"Mean spatial Bias: {np.nanmean(bias_spatial):.2f}", 
        transform=ax.transAxes, fontsize=10, verticalalignment='top', zorder=10, 
        horizontalalignment='left', color = 'red', 
        bbox={'facecolor': 'white', 'pad': 5, 'edgecolor' : 'white'})
plt.savefig(f"{graph_dir}/daily_spatial_bias_distribution.png")


# Temporal distribution
## monthly RMSE
df_rmse = pd.DataFrame({'date': dates, 'rmse_temporal': rmse_temporal})
df_rmse['month'] = pd.to_datetime(df_rmse['date']).dt.month
df_rmse['year'] = pd.to_datetime(df_rmse['date']).dt.year
rmse_monthly_mean = df_rmse.groupby('month')['rmse_temporal'].mean()
rmse_per_year = df_rmse.pivot_table(index='month', columns='year', values='rmse_temporal')
plt.figure(figsize=(10, 6))
plt.suptitle(f'baseline ({target} Evalutaion)', fontsize=16)
ax = plt.gca()
plt.title(f'Daily RMSE seasonal cycle ({period})')
plt.plot(rmse_monthly_mean.index, rmse_monthly_mean.values, label='Mean', color='red', linewidth=2)
for year in rmse_per_year.columns:
    plt.plot(rmse_per_year.index, rmse_per_year[year], label=str(year), alpha=0.3, linestyle='--')
plt.xticks(ticks=np.arange(1, 13), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=12)
plt.ylabel('RMSE (K)')
plt.xlabel('Month')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=12, ncol=2)
ax.text(0.02, 0.10, f"Mean temporal RMSE: {np.mean(rmse_temporal):.2f}", transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', horizontalalignment='left', color = 'red')
plt.tight_layout()
plt.savefig(f"{graph_dir}/daily_rmse_seasonal.png") 




