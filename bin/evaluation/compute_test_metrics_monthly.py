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
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, DomainCrop
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
from iriscc.transforms import UnPad
from iriscc.plotutils import plot_map_contour

exp = str(sys.argv[1]) # ex : exp 1
target = str(sys.argv[2]) # ex : eobs or safran
test_name = str(sys.argv[3]) # ex : mask_continents
cmip6_test = str(sys.argv[4]) # Perfect prognosis, no or cmip6, cmip6_bc


run_dir = RUNS_DIR/f'{exp}/{test_name}/lightning_logs/version_best'
checkpoint_dir = glob.glob(str(run_dir/f'checkpoints/best-checkpoint*.ckpt'))[0]
if cmip6_test == 'cmip6' or cmip6_test == 'cmip6_bc':
    test_name = f'{test_name}_{cmip6_test}'
graph_dir = GRAPHS_DIR/f'metrics/{exp}/{test_name}/'
metric_dir = METRICS_DIR/f'{exp}/mean_metrics'
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(metric_dir, exist_ok=True)


model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
model.eval()
hparams = model.hparams['hparams']
arch = hparams['model']
domain = hparams['domain']
if target == 'safran':
    domain = 'france_xy'
transforms = v2.Compose([
            MinMaxNormalisation(hparams['sample_dir'], hparams['output_norm']), 
            LandSeaMask(hparams['mask'], hparams['fill_value']),
            FillMissingValue(hparams['fill_value']),
            DomainCrop(hparams['sample_dir'], hparams['domain_crop']),
            Pad(hparams['fill_value'])
            ])
device = 'cpu'
sample_dir = hparams['sample_dir']
pp = ''
dates = DATES_TEST
if cmip6_test == 'cmip6' or cmip6_test == 'cmip6_bc':
    dates = DATES_BC_TEST_HIST
    sample_dir = DATASET_BC_DIR / f'dataset_{exp}_test_{cmip6_test}' # bc or not
    pp = f'_{cmip6_test}'


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
        condition_saf = np.isnan(y[0])

        x, y = transforms((x, y))

        x = torch.unsqueeze(x, dim=0).float()
        y_hat = model(x.to(device)).to(device)
        y_hat = y_hat.detach().cpu()

        if target == 'safran':
            unpad_func = UnPad(TARGET_SIZE)
            y, y_hat = unpad_func(y)[0].numpy(), unpad_func(y_hat[0])[0].numpy()
            condition = condition_saf
        else:
            condition = y[0] == 0. # transformed
            y = y[0,...].numpy()
            y_hat = y_hat[0,0,...].numpy()
            
        y[condition] = np.nan
        y_hat[condition] = np.nan
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
    'corr_spatial': corr_spatial}
np.savez(metric_dir/f'metrics_test_monthly_{exp}_{test_name}.npz', **d)

# Save mean values
d_mean = {'rmse_temporal_mean' : [np.mean(rmse_temporal), np.mean(rmse_temporal_summer), np.mean(rmse_temporal_winter)],
    'rmse_spatial_mean' : [np.nanmean(rmse_spatial),np.nanmean(rmse_spatial_summer), np.nanmean(rmse_spatial_winter)],
    'bias_spatial_mean' : [np.nanmean(bias_spatial),np.nanmean(bias_spatial_summer), np.nanmean(bias_spatial_winter)],
    'bias_spatial_std' : [np.nanstd(bias_spatial),np.nanstd(bias_spatial_summer), np.nanstd(bias_spatial_winter)],
    'corr_spatial_mean' : [np.mean(corr_spatial), np.mean(corr_spatial_summer), np.mean(corr_spatial_winter)],
    'corr_temporal_mean' : [np.mean(corr_temporal), np.mean(corr_temporal_summer), np.mean(corr_temporal_winter)]}

df = pd.DataFrame(d_mean, index = ['all', 'summer', 'winter'])
df.to_csv(metric_dir/f'metrics_test_mean_monthly_{exp}_{test_name}.csv')
print(df)

# Spatial distribution
## RMSE

levels = np.arange(0, 3.25, 0.25) 
colors = [
    '#a1d99b', '#41ab5d', '#006d2c',  # Vert clair -> foncé
    '#ffeda0', '#feb24c', '#d45f00',
    '#fc9272', '#de2d26', '#a50f15',   # Rouge clair -> foncé
    '#9ecae1', '#3182bd', '#08519c'
]
fig, ax = plot_map_contour(rmse_spatial,
                    domain = CONFIG[target]['domain'][domain],
                    data_projection = CONFIG[target]['data_projection'],
                    fig_projection = CONFIG[target]['fig_projection'][domain],
                    title = f'{test_name} ({target} Evalutaion)',
                    cmap=mcolors.ListedColormap(colors[:len(levels) - 1]),
                    levels=levels ,
                    var_desc='RMSE (K)')
ax.text(0.03, 0.07, f"Mean spatial RMSE: {np.nanmean(rmse_spatial):.2f}", 
        transform=ax.transAxes, fontsize=10, verticalalignment='top', zorder=10, 
        horizontalalignment='left', color = 'red', 
        bbox={'facecolor': 'white', 'pad': 5, 'edgecolor' : 'white'})
plt.savefig(f"{graph_dir}/monthly_spatial_rmse_distribution{pp}.png")

## Bias
fig, ax = plot_map_contour(bias_spatial,
                    domain = CONFIG[target]['domain'][domain],
                    data_projection = CONFIG[target]['data_projection'],
                    fig_projection = CONFIG[target]['fig_projection'][domain],
                    title = f'{test_name} ({target} Evalutaion)',
                    cmap='BrBG',
                    levels= np.linspace(-2,2,9),
                    var_desc='Bias (K)')
ax.text(0.03, 0.07, f"Mean spatial Bias: {np.nanmean(bias_spatial):.2f}", 
        transform=ax.transAxes, fontsize=10, verticalalignment='top', zorder=10, 
        horizontalalignment='left', color = 'red', 
        bbox={'facecolor': 'white', 'pad': 5, 'edgecolor' : 'white'})
plt.savefig(f"{graph_dir}/monthly_spatial_bias_distribution{pp}.png")






