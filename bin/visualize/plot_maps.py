"""
Generic plotting script for visualizing maps across different experiments.
"""

import argparse
import sys
from pathlib import Path

sys.path.append('.')

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from iriscc.datautils import Data, crop_domain_from_ds, remove_countries
from iriscc.plotutils import plot_map_contour
from iriscc.settings import CONFIG, GRAPHS_DIR, PREDICTION_DIR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize maps for a specific experiment and date")
    parser.add_argument('--exp', type=str, default='exp5', help='Experiment name (e.g., exp5)')
    parser.add_argument('--date', type=str, default='2014-12-31', help='Date to visualize (YYYY-MM-DD)')
    parser.add_argument('--simu', type=str, default='gcm', help='gcm or rcm')
    parser.add_argument('--model', type=str, default='unet', help='unet or swinunet')
    args = parser.parse_args()

    exp = args.exp
    date = pd.Timestamp(args.date)
    simu = args.simu
    model = args.model
    var = CONFIG[exp]['target_vars'][0]

    get_data = Data(CONFIG[exp]['domain'])

    # 1. Load Reference (Target)
    ds_target = get_data.get_target_dataset(target=CONFIG[exp]['target'], var=var, date=date, exp=exp)
    data_ref = ds_target[var].values
    if CONFIG[exp].get('remove_countries', False):
        data_ref = remove_countries(data_ref, domain=CONFIG[exp]['domain'])

    # 2. Load Prediction
    # Search for the prediction file
    pattern = f'{var}*historical*r1i1p1f2*{exp}_{model}_all_{simu}_bc.nc'
    pred_files = list(PREDICTION_DIR.glob(pattern))
    if not pred_files:
        print(f"Error: No prediction file found for pattern: {pattern}")
        sys.exit(1)
    
    ds_pred = xr.open_dataset(pred_files[0])
    ds_pred = ds_pred.sel(time=ds_pred.time.dt.date == date.date()).isel(time=0)
    data_pred = ds_pred[var].values

    # 3. Plotting
    levels = np.linspace(np.nanmin(data_ref), np.nanmax(data_ref), 10)
    
    # Use spatial details from CONFIG
    domain = CONFIG[exp].get('domain_xy', CONFIG[exp]['domain'])
    data_proj = CONFIG[exp]['data_projection']
    fig_proj = CONFIG[exp].get('fig_projection', ccrs.PlateCarree())

    plot_map_contour(data_ref,
                    domain = domain,
                    data_projection = data_proj,
                    fig_projection = fig_proj,
                    title = f'{var} {CONFIG[exp]["target"].upper()} (Ref) {args.date}',
                    cmap='OrRd',
                    var_desc='K',
                    levels=levels,
                    save_dir=GRAPHS_DIR/f'visualize_{exp}_ref_{args.date}.png')

    plot_map_contour(data_pred,
                    domain = domain,
                    data_projection = data_proj,
                    fig_projection = fig_proj,
                    title = f'{var} {model.upper()} Prediction {args.date}',
                    cmap='OrRd',
                    var_desc='K',
                    levels=levels,
                    save_dir=GRAPHS_DIR/f'visualize_{exp}_{model}_{args.date}.png')

    print(f"Maps saved to {GRAPHS_DIR}")
