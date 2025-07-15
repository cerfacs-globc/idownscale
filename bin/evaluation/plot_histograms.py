import sys
sys.path.append('.')

import numpy as np
import xarray as xr
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from iriscc.datautils import standardize_longitudes, crop_domain_from_ds
from iriscc.settings import CONFIG,METRICS_DIR, GRAPHS_DIR, GCM_RAW_DIR, PREDICTION_DIR, COLORS, SAFRAN_REFORMAT_DIR, EOBS_RAW_DIR, RCM_RAW_DIR
from iriscc.plotutils import plot_histogram

parser = argparse.ArgumentParser(description="Predict and plot results")
parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)') 
parser.add_argument('--simu', type=str, help='Simulation name (e.g., rcm, gcm)')  
args = parser.parse_args()

simu = args.simu

if args.exp == 'exp3':
    safran = xr.open_mfdataset(np.sort(glob.glob(str(SAFRAN_REFORMAT_DIR/f'tas*reformat.nc'))), combine='by_coords').sel(time=slice('2000', '2015'))
    tas = safran.tas.values.flatten()
    target_name = 'SAFRAN 8km'
    safran.close()
if args.exp == 'exp5':
    eobs = xr.open_dataset(EOBS_RAW_DIR/'tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc').sel(time=slice('2000', '2015'))
    tas = eobs.tas.values.flatten() + 273.15
    target_name = 'E-OBS 25km'
    eobs.close()


if simu == 'gcm':
    data = xr.open_dataset(glob.glob(str(GCM_RAW_DIR/f'CNRM-CM6-1/tas*historical_r1i1p1f2*.nc'))[0]).sel(time=slice('2000', '2015'))
    data = standardize_longitudes(data)
    data = crop_domain_from_ds(data, CONFIG[args.exp]['domain'])
    name = 'GCM 1°'

elif simu == 'rcm':
    data = xr.open_dataset(glob.glob(str(RCM_RAW_DIR/f'ALADIN_reformat/tas*historical_r1i1p1f2*.nc'))[0]).sel(time=slice('2000', '2015'))
    simu_name = 'RCM 12km'
tas_data = data.tas.values.flatten()
data.close()

data_unet = xr.open_dataset(glob.glob(str(PREDICTION_DIR/f'tas*historical_r1i1p1f2*{args.exp}_unet_all_{simu}_bc.nc'))[0]).sel(time=slice('2000', '2015'))
tas_unet = data_unet.tas.values.flatten()
data_unet.close()

data_swinunet = xr.open_dataset(glob.glob(str(PREDICTION_DIR/f'tas*historical_r1i1p1f2*{args.exp}_swinunet_all_{simu}_bc.nc'))[0]).sel(time=slice('2000', '2015'))
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