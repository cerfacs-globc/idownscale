import sys
sys.path.append('.')

import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision.transforms import v2
from torchmetrics import MeanSquaredError, PearsonCorrCoef

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue
from iriscc.settings import DATES_TEST, GRAPHS_DIR, TARGET_SIZE, RUNS_DIR, METRICS_DIR, DATASET_TEST_BC_CMIP6_DIR
from iriscc.transforms import UnPad

exp = str(sys.argv[1]) # ex : exp 1
test_name = str(sys.argv[2]) # ex : mask_continents
cmip6_test = str(sys.argv[3]) # Perfect prognosis, yes or no

run_dir = RUNS_DIR/f'{exp}/{test_name}/lightning_logs/version_best'
checkpoint_dir = run_dir/'checkpoints/best-checkpoint.ckpt'
graph_dir = GRAPHS_DIR/f'metrics/{exp}/{test_name}/'
metric_dir = METRICS_DIR/f'{exp}/mean_metrics'
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)


model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
model.eval()
hparams = model.hparams['hparams']
arch = hparams['model']
transforms = v2.Compose([
            MinMaxNormalisation(hparams['sample_dir']), 
            LandSeaMask(hparams['mask'], hparams['fill_value']),
            FillMissingValue(hparams['fill_value']),
            Pad(hparams['fill_value'])
            ])
device = 'cpu'
sample_dir = hparams['sample_dir']
pp = ''
dates = DATES_TEST
if cmip6_test == 'yes':
    test_name = f'{test_name}_cmip6'
    sample_dir = DATASET_TEST_BC_CMIP6_DIR
    pp = '_cmip6'

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
    x, y = data['x'], data['y']
    condition = np.isnan(y[0])

    x, y = transforms((x, y))

    x = torch.unsqueeze(x, dim=0).float()
    y_hat = model(x.to(device)).to(device)
    y_hat = y_hat.detach().cpu()

    unpad_func = UnPad(TARGET_SIZE)
    y, y_hat = unpad_func(y)[0].numpy(), unpad_func(y_hat[0])[0].numpy()
    y[condition] = np.nan
    y_hat[condition] = np.nan


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

    y_temporal.append(y_flat)
    y_hat_temporal.append(y_hat_flat)


y_temporal, y_hat_temporal = torch.stack(y_temporal), torch.stack(y_hat_temporal)
corr_temporal = [corr(y_hat_temporal[:,i], y_temporal[:,i]).cpu() for i in range(y_temporal.size(dim=1))]
corr_temporal = np.stack(corr_temporal)
corr_temporal_summer = [corr(y_hat_temporal[:,i], y_temporal[:,i]).cpu() for i in i_summer]
corr_temporal = np.stack(corr_temporal_summer)
corr_temporal_winter = [corr(y_hat_temporal[:,i], y_temporal[:,i]).cpu() for i in i_winter]
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

# Scalars
d = {'rmse_temporal_mean' : [np.mean(rmse_temporal), np.mean(rmse_temporal_summer), np.mean(rmse_temporal_winter)],
    'rmse_spatial_mean' : [np.nanmean(rmse_spatial),np.nanmean(rmse_spatial_summer), np.nanmean(rmse_spatial_winter)],
    'bias_spatial_mean' : [np.nanmean(bias_spatial),np.nanmean(bias_spatial_summer), np.nanmean(bias_spatial_winter)],
    'bias_spatial_std' : [np.nanstd(bias_spatial),np.nanstd(bias_spatial_summer), np.nanstd(bias_spatial_winter)],
    'corr_spatial_mean' : [np.mean(corr_spatial), np.mean(corr_spatial_summer), np.mean(corr_spatial_winter)],
    'corr_temporal_mean' : [np.mean(corr_temporal), np.mean(corr_temporal_summer), np.mean(corr_temporal_winter)]}

df = pd.DataFrame(d, index = ['all', 'summer', 'winter'])
df.to_csv(metric_dir/f'metrics_test_mean_daily_{exp}_{test_name}.csv')
print(df)

# Spatial distribution
## RMSE
plt.figure(figsize=(8, 6))
ax = plt.gca()
plt.title(f'{test_name} (SAFRAN Evalutaion)', fontsize=18)
levels = np.arange(0, 3.25, 0.25) 
colors = [
    '#a1d99b', '#41ab5d', '#006d2c',  # Vert clair -> foncé
    '#ffeda0', '#feb24c', '#d45f00',
    '#fc9272', '#de2d26', '#a50f15',   # Rouge clair -> foncé
    '#9ecae1', '#3182bd', '#08519c'
]
'''
levels = np.arange(0, 6.5, 0.5) 
colors = [
    '#a1d99b', '#41ab5d', '#006d2c',  # Vert clair -> foncé
    '#ffeda0', '#feb24c', '#d45f00',
    '#fc9272', '#de2d26', '#a50f15',   # Rouge clair -> foncé
    '#9ecae1', '#3182bd', '#08519c'
]
'''
cmap = mcolors.ListedColormap(colors[:len(levels) - 1])
cs = ax.contourf(rmse_spatial, levels = levels, cmap=cmap )
cbar = plt.colorbar(cs, ax=ax, pad=0.05)
cbar.set_label(label='RMSE (K)', size=16)
cbar.ax.tick_params(labelsize=14)
plt.axis('off')
ax.text(0.02, 0.05, f"Mean RMSE: {np.nanmean(rmse_spatial):.2f}", transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', horizontalalignment='left', color = 'red')
plt.tight_layout()
plt.savefig(f"{graph_dir}/daily_spatial_rmse_distribution{pp}.png") 

## Bias
plt.figure(figsize=(8, 6))
ax = plt.gca()
plt.title(f'{test_name} (SAFRAN Evalutaion)', fontsize=16)
cs = ax.contourf(bias_spatial, cmap='BrBG', levels= np.linspace(-1.5,1.5,13))
cbar = plt.colorbar(cs, ax=ax, pad=0.05)
cbar.set_label(label='Bias (K)', size=16)
cbar.ax.tick_params(labelsize=14)
plt.axis('off')
ax.text(0.02, 0.05, f"Mean spatial Bias: {np.nanmean(bias_spatial):.2f}", transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', horizontalalignment='left', color = 'red')
plt.tight_layout()
plt.savefig(f"{graph_dir}/daily_spatial_bias_distribution{pp}.png") 


# Temporal distribution
## monthly RMSE
df_rmse = pd.DataFrame({'date': dates, 'rmse_temporal': rmse_temporal})
df_rmse['month'] = pd.to_datetime(df_rmse['date']).dt.month
df_rmse['year'] = pd.to_datetime(df_rmse['date']).dt.year
rmse_monthly_mean = df_rmse.groupby('month')['rmse_temporal'].mean()
rmse_per_year = df_rmse.pivot_table(index='month', columns='year', values='rmse_temporal')
plt.figure(figsize=(10, 6))
plt.suptitle(f'{arch} ({test_name} config)', fontsize=16)
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
plt.savefig(f"{graph_dir}/daily_rmse_seasonal{pp}.png") 




