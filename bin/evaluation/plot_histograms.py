"""
Plot histograms of daily temperature distributions for different models and datasets.
"""

import sys
sys.path.append('.')

import numpy as np
import xarray as xr
import argparse
import glob
import matplotlib.pyplot as plt

from iriscc.datautils import standardize_longitudes, crop_domain_from_ds, Data
from iriscc.settings import CONFIG, GRAPHS_DIR, GCM_RAW_DIR, PREDICTION_DIR, COLORS, SAFRAN_REFORMAT_DIR, EOBS_RAW_DIR, RCM_RAW_DIR
from iriscc.plotutils import plot_histogram

parser = argparse.ArgumentParser(description="Predict and plot results")
parser.add_argument('--exp', type=str, default='exp5', help='Experiment name (e.g., exp5)') 
parser.add_argument('--simu', type=str, help='Simulation name (e.g., rcm, gcm)')  
args = parser.parse_args()
simu = args.simu

exp = args.exp
var = CONFIG[exp]['target_vars'][0]
get_data = Data(CONFIG[exp]['domain'])

# Load target data dynamically
ds_target = get_data.get_target_dataset(target=CONFIG[exp]['target'], var=var, date=None, exp=exp)
# Select specific period for histogram
ds_target = ds_target.sel(time=slice('2000', '2014'))
tas = ds_target[var].values.flatten()

if CONFIG[exp]['target'] == 'safran':
    target_name = 'SAFRAN 8km'
elif CONFIG[exp]['target'] == 'eobs':
    target_name = 'E-OBS 25km'
elif CONFIG[exp]['target'] == 'cerra':
    target_name = 'CERRA 5km'
else:
    target_name = CONFIG[exp]['target'].upper()

ds_target.close()


if simu == 'gcm':
    data = xr.open_dataset(next(GCM_RAW_DIR.glob('CNRM-CM6-1/tas*historical_r1i1p1f2*.nc'))).sel(time=slice('2000', '2015'))
    data = standardize_longitudes(data)
    data = crop_domain_from_ds(data, CONFIG[args.exp]['domain'])
    name = 'GCM 1°'

elif simu == 'rcm':
    data = xr.open_dataset(next(RCM_RAW_DIR.glob('ALADIN_reformat/tas*historical_r1i1p1f2*.nc'))).sel(time=slice('2000', '2015'))
    simu_name = 'RCM 12km'
tas_data = data.tas.values.flatten()
data.close()

data_unet = xr.open_dataset(next(PREDICTION_DIR.glob(f'tas*historical_r1i1p1f2*{args.exp}_unet_all_{simu}_bc.nc'))).sel(time=slice('2000', '2015'))
tas_unet = data_unet.tas.values.flatten()
data_unet.close()

data_swinunet = xr.open_dataset(next(PREDICTION_DIR.glob(f'tas*historical_r1i1p1f2*{args.exp}_swinunet_all_{simu}_bc.nc'))).sel(time=slice('2000', '2015'))
tas_swinunet = data_swinunet.tas.values.flatten()
data_swinunet.close()

labels = [target_name, simu_name, 'UNet', 'SwinUNETR']
colors = [COLORS[i] for i in labels]
tas_list = [tas, tas_data, tas_unet, tas_swinunet]

fig1, axes1 = plt.subplots(
        1, 2,
        figsize=(10,5)
    )
plot_histogram([tas_list[0], tas_list[2]], axes1[0], 
               labels=[labels[0], labels[2]], 
               colors=[colors[0],colors[2]], 
               xlabel=None)
plot_histogram([tas_list[1], tas_list[2]], axes1[1], 
               labels=[labels[1], labels[2]], 
               colors=[colors[1],colors[2]], 
               xlabel=None)
fig1.suptitle(f'Daily temperature ditribution 2000-2014 [{labels[2]}]', fontsize=16)
fig1.savefig(GRAPHS_DIR/f'metrics/{args.exp}/{args.exp}_hist_historical_unet_{simu}.png')

fig2, axes2 = plt.subplots(
        1, 2,
        figsize=(10,5)
    )
plot_histogram([tas_list[0], tas_list[3]], axes2[0], 
               labels=[labels[0], labels[3]], 
               colors=[colors[0],colors[3]], 
               xlabel=None)
plot_histogram([tas_list[1], tas_list[3]], axes2[1], 
               labels=[labels[1], labels[3]], 
               colors=[colors[1],colors[3]], 
               xlabel=None)
fig2.suptitle(f'Daily temperature ditribution 2000-2014 [{labels[3]}]', fontsize=16)
fig2.savefig(GRAPHS_DIR/f'metrics/{args.exp}/{args.exp}_hist_historical_swinunet_{simu}.png')