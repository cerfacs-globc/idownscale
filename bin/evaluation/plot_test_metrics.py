import sys
sys.path.append(".")

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from iriscc.settings import CONFIG,METRICS_DIR, GRAPHS_DIR
from iriscc.plotutils import (
    get_shared_metric_levels,
    plot_map_contour,
    plot_monthly_var_seasonal_cycle,
)

parser = argparse.ArgumentParser(description="Predict and plot results for full period")
parser.add_argument("--exp", type=str, help="Experiment name (e.g., exp1)")
parser.add_argument("--test-name", type=str, help="Test name (e.g., unet_gcm_bc)")
parser.add_argument("--scale", type=str, help="Scale (e.g., daily, monthly)")
args = parser.parse_args()
print("oks")
metrics_file = METRICS_DIR / f"{args.exp}/mean_metrics/metrics_test_{args.scale}_{args.exp}_{args.test_name}.npz"
graph_dir = Path(GRAPHS_DIR) / "metrics" / args.exp / args.test_name
graph_dir.mkdir(parents=True, exist_ok=True)

metrics_dict = dict(np.load(metrics_file, allow_pickle=True))
rmse_temporal = metrics_dict["rmse_temporal"]
rmse_spatial = metrics_dict["rmse_spatial"]
bias_spatial = metrics_dict["bias_spatial"]
dates = pd.to_datetime(metrics_dict["dates"])
if len(dates) != len(rmse_temporal):
    raise ValueError(
        f"Metric dates length {len(dates)} does not match temporal metric length {len(rmse_temporal)}"
    )
target= CONFIG[args.exp]["target"]

h, w  = CONFIG[args.exp]["shape"]
if CONFIG[args.exp]["target"] == "safran":
    domain = CONFIG[args.exp]["domain_xy"]
else:
     domain = CONFIG[args.exp]["domain"]
if args.test_name.endswith("pp"):
    target = "rcm"
rmse_spatial = rmse_spatial.reshape(h, w)
bias_spatial = bias_spatial.reshape(h, w)
print(np.nanmax(bias_spatial), np.nanmax(rmse_spatial))

# Spatial distribution
## RMSE
levels = get_shared_metric_levels("rmse", args.scale)
rmse_palette = [
    "#9ecae1", "#3182bd", "#08519c",
    "#a1d99b", "#41ab5d", "#006d2c",  # Vert clair -> foncé
    "#ffeda0", "#feb24c", "#d45f00",
    "#fc9272", "#de2d26", "#a50f15"   # Rouge clair -> foncé

]
rmse_cmap = mcolors.LinearSegmentedColormap.from_list(
    "shared_rmse",
    rmse_palette,
    N=len(levels) - 1,
)
fig, ax = plot_map_contour(rmse_spatial,
                    domain = domain,
                    data_projection = CONFIG[args.exp]["data_projection"],
                    fig_projection = CONFIG[args.exp]["fig_projection"],
                    title = f"{args.scale} {args.test_name} ({target})",
                    cmap=rmse_cmap,
                    levels=levels ,
                    extend="max",
                    var_desc="RMSE (K)")
ax.text(0.03, 0.07, f"Mean spatial RMSE: {np.nanmean(rmse_spatial):.2f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top", zorder=10,
        horizontalalignment="left", color = "red",
        bbox={"facecolor": "white", "pad": 5, "edgecolor" : "white"})
plt.savefig(graph_dir / f"{args.scale}_spatial_rmse_distribution_{args.test_name}.png")

## Bias
levels = get_shared_metric_levels("bias", args.scale)
fig, ax = plot_map_contour(bias_spatial,
                    domain = domain,
                    data_projection = CONFIG[args.exp]["data_projection"],
                    fig_projection = CONFIG[args.exp]["fig_projection"],
                    title = f"{args.scale} {args.test_name} ({target})",
                    cmap="BrBG",
                    levels=levels,
                    extend="both",
                    var_desc="Bias (K)")
ax.text(0.03, 0.07, f"Mean spatial Bias: {np.nanmean(bias_spatial):.2f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top", zorder=10,
        horizontalalignment="left", color = "red",
        bbox={"facecolor": "white", "pad": 5, "edgecolor" : "white"})
plt.savefig(graph_dir / f"{args.scale}_spatial_bias_distribution_{args.test_name}.png")

if args.scale == "monthly":
    plt.figure()
    print(rmse_temporal.dtype)
    plt.plot(dates, list(rmse_temporal), label="RMSE", color="blue")
    plt.savefig(Path(GRAPHS_DIR) / "test1.png")


# Temporal distribution
## monthly RMSE
plot_monthly_var_seasonal_cycle(var_temporal=rmse_temporal,
                                dates=dates,
                                title=f"{args.scale} {args.test_name} ({target})",
                                var_desc="RMSE (K)",
                                save_dir=graph_dir / f"{args.scale}_rmse_seasonal_{args.test_name}.png")
