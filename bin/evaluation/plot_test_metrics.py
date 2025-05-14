import sys
sys.path.append('.')

import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from iriscc.settings import CONFIG,METRICS_DIR, GRAPHS_DIR
from iriscc.plotutils import plot_map_contour, plot_monthly_var_seasonal_cycle
from iriscc.datautils import datetime_period_to_string

parser = argparse.ArgumentParser(description="Predict and plot results for full period")
parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')   
parser.add_argument('--target', type=str, help='Target data (e.g., safran, eobs)')
parser.add_argument('--test-name', type=str, help='Test name (e.g., unet_cmip6_bc)')
parser.add_argument('--domain', type=str, help='Domain name (e.g., france_xy)', default='france_xy')
args = parser.parse_args()

metrics_file = METRICS_DIR / f'{args.exp}/mean_metrics/metrics_test_daily_{args.exp}_{args.test_name}.npz'
graph_dir = GRAPHS_DIR/f'metrics/{args.exp}/{args.test_name}/'

metrics_dict = dict(np.load(metrics_file, allow_pickle=True))
rmse_temporal = metrics_dict['rmse_temporal']
rmse_spatial = metrics_dict['rmse_spatial']
bias_spatial = metrics_dict['bias_spatial']
dates = metrics_dict['dates']
period = datetime_period_to_string(dates)

h, w  = CONFIG[args.target]['shape'][args.domain]

rmse_spatial = rmse_spatial.reshape(h, w)
bias_spatial = bias_spatial.reshape(h, w)

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
                    domain = CONFIG[args.target]['domain'][args.domain],
                    data_projection = CONFIG[args.target]['data_projection'],
                    fig_projection = CONFIG[args.target]['fig_projection'][args.domain],
                    title = f'{args.test_name} {period} ({args.target})',
                    cmap=mcolors.ListedColormap(colors[:len(levels) - 1]),
                    levels=levels ,
                    var_desc='RMSE (K)')
ax.text(0.03, 0.07, f"Mean spatial RMSE: {np.nanmean(rmse_spatial):.2f}", 
        transform=ax.transAxes, fontsize=10, verticalalignment='top', zorder=10, 
        horizontalalignment='left', color = 'red', 
        bbox={'facecolor': 'white', 'pad': 5, 'edgecolor' : 'white'})
plt.savefig(f"{graph_dir}/daily_spatial_rmse_distribution_{args.test_name}.png")

## Bias
levels = np.linspace(-2,2,9)
fig, ax = plot_map_contour(bias_spatial,
                    domain = CONFIG[args.target]['domain'][args.domain],
                    data_projection = CONFIG[args.target]['data_projection'],
                    fig_projection = CONFIG[args.target]['fig_projection'][args.domain],
                    title = f'{args.test_name} {period} ({args.target})',
                    cmap='BrBG',
                    levels=levels,
                    var_desc='Bias (K)')
ax.text(0.03, 0.07, f"Mean spatial Bias: {np.nanmean(bias_spatial):.2f}", 
        transform=ax.transAxes, fontsize=10, verticalalignment='top', zorder=10, 
        horizontalalignment='left', color = 'red', 
        bbox={'facecolor': 'white', 'pad': 5, 'edgecolor' : 'white'})
plt.savefig(f"{graph_dir}/daily_spatial_bias_distribution_{args.test_name}.png")


# Temporal distribution
## monthly RMSE
plot_monthly_var_seasonal_cycle(var_temporal=rmse_temporal, 
                                dates=dates, 
                                title=f'{args.test_name} {period} ({args.target})', 
                                var_desc='RMSE (K)', 
                                save_dir=f"{graph_dir}/daily_rmse_seasonal_{args.test_name}.png")



