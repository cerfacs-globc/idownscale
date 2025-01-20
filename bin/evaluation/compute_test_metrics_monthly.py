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
from iriscc.settings import DATASET_EXP1_DIR, DATES_TEST, GRAPHS_DIR, TARGET_SIZE, RUNS_DIR, METRICS_DIR, DATASET_EXP1_30Y_DIR
from iriscc.transforms import UnPad

exp = str(sys.argv[1]) # ex : exp 1
test_name = str(sys.argv[2]) # ex : mask_continents
version = str(sys.argv[3])
run_dir = RUNS_DIR/f'{exp}/{test_name}/lightning_logs/version_{version}'
checkpoint_dir = run_dir/'checkpoints/best-checkpoint.ckpt'
graph_dir = GRAPHS_DIR/f'metrics/{exp}/{test_name}/'
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

monthly_indices = defaultdict(list)
monthly_y = defaultdict(list)
monthly_y_hat = defaultdict(list)

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

startdate = DATES_TEST[0].date().strftime('%d/%m/%Y')
enddate = DATES_TEST[-1].date().strftime('%d/%m/%Y')
period = f'{startdate} - {enddate}'

# Regroupement des indices par mois
for i, date in enumerate(DATES_TEST):
    print(date)
    month_id = date.strftime('%Y/%m')
    monthly_indices[month_id].append(i)

    if date.month in [6, 7, 8]:
        i_summer.append(i)
    if date.month in [1, 2, 12]:
        i_winter.append(i)

    # Chargement des données
    date_str = date.date().strftime('%Y%m%d')
    sample = glob.glob(str(hparams['sample_dir'] / f'sample_{date_str}.npz'))[0]
    data = dict(np.load(sample, allow_pickle=True))
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

    # Regroupement par mois
    monthly_y[month_id].append(y)
    monthly_y_hat[month_id].append(y_hat)

# Calcul des moyennes mensuelles et des métriques
for month_id, indices in monthly_indices.items():
    y_monthly = np.nanmean(monthly_y[month_id], axis=0)
    y_hat_monthly = np.nanmean(monthly_y_hat[month_id], axis=0)

    # Calcul des erreurs
    error = y_hat_monthly - y_monthly
    error_squared = error ** 2

    if len(rmse_spatial) == 0:
        rmse_spatial = error_squared
        bias_spatial = error
    else:
        rmse_spatial += error_squared
        bias_spatial += error

    # Saisons
    if month_id.split('/')[1] in ['06', '07', '08']:
        if len(rmse_spatial_summer) == 0:
            rmse_spatial_summer = error_squared
            bias_spatial_summer = error
        else:
            rmse_spatial_summer += error_squared
            bias_spatial_summer += error

    elif month_id.split('/')[1] in ['12', '01', '02']:
        if len(rmse_spatial_winter) == 0:
            rmse_spatial_winter = error_squared
            bias_spatial_winter = error
        else:
            rmse_spatial_winter += error_squared
            bias_spatial_winter += error

    # Conversion en tensors pour calcul des métriques temporelles
    y_tensor = torch.tensor(y_monthly)
    y_hat_tensor = torch.tensor(y_hat_monthly)
    y_flat, y_hat_flat = y_tensor[~torch.isnan(y_tensor)], y_hat_tensor[~torch.isnan(y_hat_tensor)]

    # Calcul des métriques temporelles
    rmse_value = rmse(y_hat_flat, y_flat).item()
    corr_value = corr(y_hat_flat - y_hat_flat.mean(), y_flat - y_flat.mean()).item()

    rmse_temporal.append(rmse_value)
    corr_spatial.append(corr_value)

# Calcul final des moyennes et des écarts-types
rmse_spatial = np.sqrt(rmse_spatial / len(DATES_TEST))
rmse_spatial_summer = np.sqrt(rmse_spatial_summer / len(i_summer))
rmse_spatial_winter = np.sqrt(rmse_spatial_winter / len(i_winter))
bias_spatial = bias_spatial / len(DATES_TEST)
bias_spatial_summer = bias_spatial_summer / len(i_summer)
bias_spatial_winter = bias_spatial_winter / len(i_winter)

rmse_temporal_summer = [rmse_temporal[i] for i in i_summer]
rmse_temporal_winter = [rmse_temporal[i] for i in i_winter]
corr_spatial_summer = [corr_spatial[i] for i in i_summer]
corr_spatial_winter = [corr_spatial[i] for i in i_winter]

# Scalars
d = {
    'rmse_temporal_mean': [np.mean(rmse_temporal), np.mean(rmse_temporal_summer), np.mean(rmse_temporal_winter)],
    'rmse_spatial_mean': [np.nanmean(rmse_spatial), np.nanmean(rmse_spatial_summer), np.nanmean(rmse_spatial_winter)],
    'bias_spatial_mean': [np.nanmean(bias_spatial), np.nanmean(bias_spatial_summer), np.nanmean(bias_spatial_winter)],
    'bias_spatial_std': [np.nanstd(bias_spatial), np.nanstd(bias_spatial_summer), np.nanstd(bias_spatial_winter)],
    'corr_spatial_mean': [np.mean(corr_spatial), np.mean(corr_spatial_summer), np.mean(corr_spatial_winter)],
    'corr_temporal_mean': [np.mean(rmse_temporal), np.mean(rmse_temporal_summer), np.mean(rmse_temporal_winter)],
}

df = pd.DataFrame(d, index=['all', 'summer', 'winter'])
df.to_csv(metric_dir / f'metrics_test_mean_{exp}_{test_name}.csv')
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
df_rmse = pd.DataFrame({'date': DATES_TEST, 'rmse_temporal': rmse_temporal})
df_rmse['month'] = pd.to_datetime(df_rmse['date']).dt.month
df_rmse['year'] = pd.to_datetime(df_rmse['date']).dt.year
rmse_monthly_mean = df_rmse.groupby('month')['rmse_temporal'].mean()
rmse_per_year = df_rmse.pivot_table(index='month', columns='year', values='rmse_temporal')
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
ax.text(0.02, 0.10, f"Mean temporal RMSE: {np.mean(rmse_temporal):.2f}", transform=ax.transAxes, fontsize=12, 
        verticalalignment='top', horizontalalignment='left', color = 'red')
plt.tight_layout()
plt.savefig(f"{graph_dir}/monthly_rmse_cycle.png") 


'''
