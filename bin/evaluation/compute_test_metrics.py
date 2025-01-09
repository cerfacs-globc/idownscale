import sys
sys.path.append('.')

import os
import glob
import torch
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchmetrics import MeanSquaredError, PearsonCorrCoef

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue
from iriscc.settings import DATASET_EXP1_DIR, DATES_TEST, GRAPHS_DIR, TARGET_SIZE, RUNS_DIR, METRICS_DIR
from iriscc.transforms import UnPad

exp = str(sys.argv[1]) # ex : exp 1
test_name = str(sys.argv[2]) # ex : mask_continents
version = str(sys.argv[3])
run_dir = RUNS_DIR/f'{exp}/{test_name}/lightning_logs/version_{version}'
checkpoint_dir = run_dir/'checkpoints/best-checkpoint.ckpt'
graph_dir = GRAPHS_DIR/f'metrics/{exp}/{test_name}/version_{version}'
metric_dir = METRICS_DIR/f'{exp}/mean_metrics'
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)


model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir)
model.eval()
hparams = model.hparams['hparams']
arch = hparams['model']
transforms = v2.Compose([
            MinMaxNormalisation(), 
            LandSeaMask(hparams['mask'], hparams['fill_value']),
            FillMissingValue(hparams['fill_value']),
            Pad(hparams['fill_value'])
            ])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

rmse = MeanSquaredError(squared=False).to(device)
corr = PearsonCorrCoef().to(device)

rmse_daily = []  
rmse_spatial = []  
bias_spatial = []
corr_spatial = []
y_temporal = []
y_hat_temporal = []
corr_temporal = []

startdate = DATES_TEST[0].date().strftime('%d/%m/%Y')
enddate = DATES_TEST[-1].date().strftime('%d/%m/%Y')
period = f'{startdate} - {enddate}'

for date in DATES_TEST:
    print(date)
    date_str = date.date().strftime('%Y%m%d')
    sample = glob.glob(str(DATASET_EXP1_DIR/f'sample_{date_str}.npz'))[0]
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

    ## temporal metrics
    y, y_hat = torch.tensor(y), torch.tensor(y_hat)
    y_flat, y_hat_flat = y[~torch.isnan(y)].to(device), y_hat[~torch.isnan(y_hat)].to(device)
    rmse_value = rmse(y_hat_flat, y_flat).item()
    corr_value = corr(y_hat_flat - torch.mean(y_hat_flat), y_flat - torch.mean(y_flat)).item()
    rmse_daily.append(rmse_value)
    corr_spatial.append(corr_value)

    y_temporal.append(y_flat)
    y_hat_temporal.append(y_hat_flat)

y_temporal, y_hat_temporal = torch.stack(y_temporal), torch.stack(y_hat_temporal)
corr_temporal = [corr(y_hat_temporal[:,i], y_temporal[:,i]).cpu() for i in range(y_temporal.size(dim=1))]
corr_temporal = np.stack(corr_temporal)
rmse_spatial = np.sqrt(rmse_spatial / len(DATES_TEST))
bias_spatial = bias_spatial / len(DATES_TEST)

# Scalars
d= {'rmse_daily_mean' : [np.mean(rmse_daily)],
    'rmse_spatial_mean' : [np.nanmean(rmse_spatial)],
    'bias_spatial_mean' : [np.nanmean(bias_spatial)],
    'corr_spatial_mean' : [np.mean(corr_spatial)],
    'corr_temporal_mean' : [np.mean(corr_temporal)]}

df = pd.DataFrame(d)
df.to_csv(metric_dir/f'metrics_test_mean_{exp}_{test_name}.csv')
print(df)

'''
# Spatial distribution
## RMSE
plt.figure(figsize=(8, 6))
plt.suptitle(f'{arch} ({test_name} config)', fontsize=16)
ax = plt.gca()
plt.title(f'Mean RMSE spatial distribution ({period})')
plt.imshow(np.flip(rmse_spatial, axis=0), cmap='OrRd', vmin=0, vmax=6)
plt.colorbar(label='RMSE (K)')
plt.axis('off')
ax.text(0.02, 0.05, f"Mean spatial RMSE: {np.nanmean(rmse_spatial):.2f}", transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', horizontalalignment='left', color = 'red')
plt.savefig(f"{graph_dir}/spatial_rmse_distribution.png") 

## Bias
plt.figure(figsize=(8, 6))
plt.suptitle(f'{arch} ({test_name} config)', fontsize=16)
ax = plt.gca()
plt.title(f'Mean bias spatial distribution ({period})')
plt.imshow(np.flip(bias_spatial, axis=0), cmap='BrBG', vmin=-5, vmax=5)
plt.colorbar(label='Bias (K)')
plt.axis('off')
ax.text(0.02, 0.05, f"Mean spatial Bias: {np.nanmean(bias_spatial):.2f}", transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', horizontalalignment='left', color = 'red')
plt.savefig(f"{graph_dir}/spatial_bias_distribution.png") 

# Temporal distribution
## monthly RMSE
df_rmse = pd.DataFrame({'date': DATES_TEST, 'rmse_daily': rmse_daily})
df_rmse['month'] = pd.to_datetime(df_rmse['date']).dt.month
df_rmse['year'] = pd.to_datetime(df_rmse['date']).dt.year
rmse_monthly_mean = df_rmse.groupby('month')['rmse_daily'].mean()
rmse_per_year = df_rmse.pivot_table(index='month', columns='year', values='rmse_daily')
plt.figure(figsize=(10, 6))
plt.suptitle(f'{arch} ({test_name} config)', fontsize=16)
ax = plt.gca()
plt.title(f'RMSE seasonal cycle ({period})')
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
ax.text(0.02, 0.10, f"Mean temporal RMSE: {np.mean(rmse_daily):.2f}", transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', horizontalalignment='left', color = 'red')
plt.tight_layout()
plt.savefig(f"{graph_dir}/monthly_rmse_cycle.png") 

'''


