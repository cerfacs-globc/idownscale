import sys
sys.path.append('.')

import glob
import pandas as pd
import xarray as xr
import argparse
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


from iriscc.settings import GCM_RAW_DIR, PREDICTION_DIR, CONFIG, GRAPHS_DIR, COLORS, RCM_RAW_DIR
from iriscc.plotutils import plot_test, plot_histogram
from iriscc.datautils import standardize_longitudes, crop_domain_from_ds, reformat_as_target

def compute_variability(data):
    var = [data[i,:,:] - data[i-1,:,:] for i in range(data.shape[0])]
    var_temporal = np.nanmean(np.abs(var), axis= (1,2))
    return var_temporal

def plot_variability(fig, axes, df_var_temporal, periods, labels, colors):
    sns.boxplot(x='period', y='Variability', hue='label', data=df_var_temporal, ax=axes[0], palette=colors, gap=0.1)
    axes[0].set_title("Temporal variability")
    axes[0].set_xlabel(None)
    axes[0].legend()

    dates = pd.date_range(*(pd.to_datetime([f'{periods[-2]}-01-01', f'{periods[-1]}-12-31'])))
    dfs = pd.DataFrame.from_dict({'dates' : dates,
                        labels[0]: 
                            df_var_temporal[(df_var_temporal['period'] == f'{periods[-2]} - {periods[-1]}') & 
                             (df_var_temporal['label'] == labels[0])]['Variability'].values,
                        labels[1]: 
                            df_var_temporal[(df_var_temporal['period'] == f'{periods[-2]} - {periods[-1]}') & 
                             (df_var_temporal['label'] == labels[1])]['Variability'].values,
                        labels[2]: 
                            df_var_temporal[(df_var_temporal['period'] == f'{periods[-2]} - {periods[-1]}') & 
                             (df_var_temporal['label'] == labels[2])]['Variability'].values})
    dfs = dfs.groupby(dfs['dates'].dt.year).mean()
    for i, label in enumerate(labels):
        axes[1].plot(dfs['dates'], 
                    dfs[label], 
                    '^-',
                    label=label,
                    color=colors[i],
                    linewidth=2.5)
    axes[1].set_title("Variability annual mean")
    axes[1].set_ylabel("Variability (K)")
    return fig, axes




if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Predict and plot results")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')   
    parser.add_argument('--ssp', type=str, help='Scenario name (e.g., ssp126, ssp585)')
    parser.add_argument('--simu', type=str, help='Simulation type (e.g., gcm, rcm)', default='gcm') 
    args = parser.parse_args()
    exp = args.exp
    ssp = args.ssp
    simu = args.simu

    periods = ['2015', '2040', '2070', '2100']   

    output_projection = CONFIG[exp]['fig_projection']
    inputs_projections = [ccrs.PlateCarree(), 
                        CONFIG[exp]['data_projection'], 
                        CONFIG[exp]['data_projection']]
    if CONFIG[exp]['target']=='safran':
        domain = [CONFIG[exp]['domain'],
                    CONFIG[exp]['domain_xy'],
                    CONFIG[exp]['domain_xy']]
    else:
        domain = [CONFIG[exp]['domain'],
                    CONFIG[exp]['domain'],
                    CONFIG[exp]['domain']]
    if simu == 'rcm':
        simu_name = 'RCM 12km'
        dir = RCM_RAW_DIR / 'ALADIN_reformat'
    else:
        simu_name = 'GCM 1°'
        dir = GCM_RAW_DIR / 'CNRM-CM6-1'
    labels = [simu_name, 'UNet', 'SwinUNETR']
    colors = [COLORS[i] for i in labels]
    colors_map = ['white', 'yellow', 'orange', 'red', 'black']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors_map, N=12)

    var_spatial = []
    var_temporal = []

    fig1, axes1 = plt.subplots(
        3, 3,
        figsize=(15, 10),
        subplot_kw={'projection': output_projection}
    )

    fig2, axes2 = plt.subplots(
        1, 3,
        figsize=(15,5)
    )
    fig3, axes3 = plt.subplots(
        1, 3,
        figsize=(15,5)
    )

    ## Réference 1980-2010
    data_ref = xr.open_mfdataset(glob.glob(str(dir/f'tas*historical_r1i1p1f2*.nc'))).sel(time=slice('1980', '2010'))
    data_ref = standardize_longitudes(data_ref)
    data_ref = crop_domain_from_ds(data_ref, CONFIG[exp]['domain'])
    data_ref = data_ref.mean(dim='time')
    tas_ref = data_ref.tas.values # mean spatial reference tas
    data_ref.close()

    unet_ref = xr.open_mfdataset(glob.glob(str(PREDICTION_DIR/f'tas*historical_r1i1p1f2*{exp}_unet_all_{simu}_bc.nc'))).sel(time=slice('1980', '2010'))
    unet_ref = unet_ref.mean(dim='time')
    tas_unet_ref = unet_ref.tas.values
    unet_ref.close()

    swinunet_ref = xr.open_mfdataset(glob.glob(str(PREDICTION_DIR/f'tas*historical_r1i1p1f2*{exp}_swinunet_all_{simu}_bc.nc'))).sel(time=slice('1980', '2010'))
    swinunet_ref = swinunet_ref.mean(dim='time')
    tas_swinunet_ref = swinunet_ref.tas.values
    swinunet_ref.close()

    for i in range(len(periods)-1):

        # GET DATA
        file = glob.glob(str(dir/f'tas*{ssp}_r1i1p1f2*.nc'))[0]
        data = xr.open_dataset(file).sel(time=slice(periods[i], periods[i+1]))
        data = standardize_longitudes(data)
        data = crop_domain_from_ds(data, CONFIG[exp]['domain'])
        tas_data = data.tas.values
        data.close()
        
        file = glob.glob(str(PREDICTION_DIR/f'tas*{ssp}_r1i1p1f2*{exp}_unet_all_{simu}_bc.nc'))[0]
        unet = xr.open_dataset(file).sel(time=slice(periods[i], periods[i+1]))
        tas_unet = unet.tas.values
        unet.close()

        file = glob.glob(str(PREDICTION_DIR/f'tas*{ssp}_r1i1p1f2*{exp}_swinunet_all_{simu}_bc.nc'))[0]
        swinunet = xr.open_dataset(file).sel(time=slice(periods[i], periods[i+1]))
        tas_swinunet = swinunet.tas.values
        swinunet.close()

        l = [tas_data, tas_unet, tas_swinunet]
        tref = [tas_ref, tas_unet_ref, tas_swinunet_ref]

        ## PLOT CHANGES MAPS
        for col in range(3):
            ax1 = axes1[i, col]
            data_proj = inputs_projections[col]
            cs = ax1.contourf(
                np.nanmean(l[col], axis=0) - tref[col],
                extent=domain[col],
                transform=data_proj,
                cmap=custom_cmap,
                levels=np.linspace(0, 6, 13),
            )
            
            ax1.set_extent([-5., 11., 41., 51.], crs=ccrs.PlateCarree())
            ax1.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
            ax1.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, edgecolor='gray', zorder=10)
            cbar = plt.colorbar(cs, ax=ax1, pad=0.05, shrink=0.8)
            cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label(label='K', size=14, labelpad=5)
            if col == 0:
                ax1.text(-0.07, 0.55, f'{periods[i]} - {periods[i+1]}', va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=ax1.transAxes, fontsize=14)


            ## PLOT VARIABILITY
            var_t = compute_variability(l[col])
            var_temporal.extend([
                {'period': f'{periods[i]} - {periods[i+1]}', 'Variability': val, 'label': labels[col]}
                for val in var_t.flatten()
            ])
            
        
        ## PLOT HISTOGRAMS 
        plot_histogram([tas_data.flatten(), tas_unet.flatten()], 
                       axes2[i], 
                       [labels[0], labels[1]], 
                       [colors[0], colors[1]], 
                       f'{periods[i]} - {periods[i+1]}')
        plot_histogram([tas_data.flatten(), tas_swinunet.flatten()], 
                       axes3[i], 
                       [labels[0], labels[2]], 
                       [colors[0], colors[2]], 
                       f'{periods[i]} - {periods[i+1]}')

    for ax, col in zip(axes1[0], labels):
        ax.set_title(col, fontsize=14)

    
    fig1.suptitle(f'Temperature changes for {ssp} (Reference : 1980-2010)', fontsize=16)
    fig1.tight_layout(rect=[0.05, 0, 1, 1], pad = 2)
    fig1.savefig(GRAPHS_DIR/f'metrics/{exp}/{exp}_spatial_futur_trend_{ssp}_{simu}.png')

    fig2.suptitle(f'Daily temperature ditribution for {ssp} {periods[0]} - {periods[-1]} [UNet]', fontsize=16)
    fig2.savefig(GRAPHS_DIR/f'metrics/{exp}/{exp}_hist_futur_trend_{ssp}_unet_{simu}.png')

    fig3.suptitle(f'Daily temperature ditribution for {ssp} {periods[0]} - {periods[-1]} [SwinUNETR]', fontsize=16)
    fig3.savefig(GRAPHS_DIR/f'metrics/{exp}/{exp}_hist_futur_trend_{ssp}_swinunet_{simu}.png')

    df_var_temporal = pd.DataFrame(var_temporal)

    fig4, axes4 = plt.subplots(2, 1, figsize=(10, 6))
    fig4, axes4 =  plot_variability(fig4, axes4, df_var_temporal, periods, labels, colors)
    fig4.suptitle(f'Daily temperature variability for {ssp}', fontsize=16)
    fig4.tight_layout()
    fig4.savefig(GRAPHS_DIR/f'metrics/{exp}/{exp}_variability_futur_trend_{ssp}_{simu}.png')


