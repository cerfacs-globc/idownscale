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
from iriscc.settings import CONFIG,METRICS_DIR, GRAPHS_DIR, CMIP6_RAW_DIR, PREDICTION_DIR, COLORS, SAFRAN_REFORMAT_DIR
from iriscc.plotutils import plot_histogram

parser = argparse.ArgumentParser(description="Predict and plot results")
parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')   
args = parser.parse_args()




gcm = xr.open_dataset(glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*historical_r1i1p1f2*.nc'))[0]).sel(time=slice('2000', '2015'))
gcm = standardize_longitudes(gcm)
gcm = crop_domain_from_ds(gcm, CONFIG['safran']['domain']['france'])
tas = gcm.tas.values.flatten()
gcm.close()

gcm_unet = xr.open_dataset(glob.glob(str(PREDICTION_DIR/f'tas*historical_r1i1p1f2*{args.exp}_unet_all_cmip6_bc.nc'))[0]).sel(time=slice('2000', '2015'))
tas_unet = gcm_unet.tas.values.flatten()
gcm_unet.close()

gcm_swinunet = xr.open_dataset(glob.glob(str(PREDICTION_DIR/f'tas*historical_r1i1p1f2*{args.exp}_swinunet_all_cmip6_bc.nc'))[0]).sel(time=slice('2000', '2015'))
tas_swinunet = gcm_swinunet.tas.values.flatten()
gcm_swinunet.close()

safran = xr.open_mfdataset(np.sort(glob.glob(str(SAFRAN_REFORMAT_DIR/f'tas*reformat.nc'))), combine='by_coords').sel(time=slice('2000', '2015'))
tas_saf = safran.tas.values.flatten()
safran.close()


labels = ['SAFRAN 8km', 'GCM 1°', 'UNet', 'SwinUNETR']
colors = [COLORS[i] for i in labels]
tas_list = [tas_saf, tas, tas_unet, tas_swinunet]

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
fig1.savefig(GRAPHS_DIR/f'metrics/{args.exp}/{args.exp}_hist_historical_unet.png')


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
fig2.savefig(GRAPHS_DIR/f'metrics/{args.exp}/{args.exp}_hist_historical_swinunet.png')