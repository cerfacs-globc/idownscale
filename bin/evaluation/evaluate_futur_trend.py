import sys
sys.path.append('.')

import glob
import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


from iriscc.settings import CMIP6_RAW_DIR, PREDICTION_DIR, CONFIG, GRAPHS_DIR, COLORS, TARGET_SAFRAN_FILE
from iriscc.plotutils import plot_test
from iriscc.datautils import standardize_longitudes, crop_domain_from_ds, reformat_as_target

def compute_variability(data):
    var = [data[i,:,:] - data[i-1,:,:] for i in range(data.shape[0])]
    var_spatial = np.nanmean(var, axis=0)
    var_temporal = np.nanmean(var, axis= (1,2))
    return var_spatial, var_temporal

def plot_variability(fig, axes, df_var_spatial, df_var_temporal, periods, labels, colors):
    sns.boxplot(x='period', y='Variability', hue='label', data=df_var_spatial, ax=axes[0], palette=colors, gap=0.1)
    axes[0].set_ylim(-0.01, 0.01)
    axes[0].set_xlabel(None)
    axes[0].set_title("Spatial variability")
    axes[0].legend()

    sns.boxplot(x='period', y='Variability', hue='label', data=df_var_temporal, ax=axes[1], legend=False, palette=colors, gap=0.1)
    axes[1].set_title("Temporal variability")
    axes[1].set_xlabel(None)

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
        axes[2].plot(dfs['dates'], 
                    dfs[label], 
                    '^-',
                    label=label,
                    color=colors[i],
                    linewidth=2.5)
    axes[2].set_title("Temperature variability annual mean")
    axes[2].set_ylabel("Variability")
    return fig, axes

if __name__=='__main__':

    exp = str(sys.argv[1])
    ssp = str(sys.argv[3])

    periods = ['2015', '2040', '2070', '2100']   

    output_projection = CONFIG['safran']['fig_projection']['france_xy']
    inputs_projections = [ccrs.PlateCarree(), 
                        CONFIG['safran']['data_projection'], 
                        CONFIG['safran']['data_projection']]
    domain = [CONFIG['safran']['domain']['france'],
                CONFIG['safran']['domain']['france_xy'],
                CONFIG['safran']['domain']['france_xy']]
    labels = ['GCM 1°', 'UNet', 'SwinUNETR']
    colors = [COLORS[i] for i in labels]

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

    ## Réference 1980-2014
    gcm_ref = xr.open_dataset(glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*historical_r1i1p1f2*.nc'))[0]).sel(time=slice('1980', '2014'))
    gcm_ref = standardize_longitudes(gcm_ref)
    gcm_ref = crop_domain_from_ds(gcm_ref, CONFIG['safran']['domain']['france'])
    gcm_ref = gcm_ref.mean(dim='time')
    tas_ref = gcm_ref.tas.values # mean spatial reference tas
    gcm_ref_ia = reformat_as_target(gcm_ref, 
                                   TARGET_SAFRAN_FILE, 
                                   method='conservative_normed', 
                                   mask=True,
                                   domain=CONFIG['safran']['domain']['france'])
    tas_ref_ia = gcm_ref_ia.tas.values
    gcm_ref.close()
    
    for i in range(len(periods)-1):

        # GET DATA
        file = glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*{ssp}_r1i1p1f2*.nc'))[0]
        gcm = xr.open_dataset(file).sel(time=slice(periods[i], periods[i+1]))
        gcm = standardize_longitudes(gcm)
        gcm = crop_domain_from_ds(gcm, CONFIG['safran']['domain']['france'])
        tas_gcm = gcm.tas.values
        gcm = gcm.mean(dim=['lat', 'lon'])
        tas_gcm_mean = gcm.resample(time='1ME', skipna=True).mean().tas.values
        gcm.close()
        
        file = glob.glob(str(PREDICTION_DIR/f'tas*{ssp}_r1i1p1f2*{exp}_unet_all_cmip6.nc'))[0]
        gcm_ia = xr.open_dataset(file).sel(time=slice(periods[i], periods[i+1]))
        tas_gcm_ia = gcm_ia.tas.values
        gcm_ia = gcm_ia.mean(dim=['x', 'y'])
        tas_gcm_ia_mean = gcm_ia.resample(time='1ME', skipna=True).mean().tas.values
        gcm_ia.close()

        file = glob.glob(str(PREDICTION_DIR/f'tas*{ssp}_r1i1p1f2*{exp}_swinunet_all_cmip6.nc'))[0]
        gcm_ia_bc = xr.open_dataset(file).sel(time=slice(periods[i], periods[i+1]))
        tas_gcm_ia_bc = gcm_ia_bc.tas.values
        gcm_ia_bc = gcm_ia_bc.mean(dim=['x', 'y'])
        tas_gcm_ia_bc_mean = gcm_ia_bc.resample(time='1ME', skipna=True).mean().tas.values
        gcm_ia_bc.close()

        l = [tas_gcm, tas_gcm_ia, tas_gcm_ia_bc]
        tref = [tas_ref, tas_ref_ia, tas_ref_ia]

        ## PLOT CHANGES MAPS
        for col in range(3):
            ax1 = axes1[i, col]
            data_proj = inputs_projections[col]
            img = ax1.imshow(
                np.nanmean(l[col], axis=0) - tref[col],
                extent=domain[col],
                transform=data_proj,
                origin='lower',
                cmap='bwr',
                vmin=-10,
                vmax=10
            )

            ax1.set_extent([-5., 11., 41., 51.], crs=ccrs.PlateCarree())
            ax1.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
            ax1.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, edgecolor='gray', zorder=10)
            cbar = plt.colorbar(img, ax=ax1, pad=0.05, shrink=0.8)
            cbar.set_label(label='K', size=12, labelpad=2.5)
            if col == 0:
                ax1.text(-0.07, 0.55, f'{periods[i]} - {periods[i+1]}', va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor',
                transform=ax1.transAxes, fontsize=14)


            ## PLOT VARIABILITY
            var_s, var_t = compute_variability(l[col])
            var_spatial.extend([
                {'period': f'{periods[i]} - {periods[i+1]}', 'Variability': val, 'label': labels[col]}
                for val in var_s.flatten()
            ])

            var_temporal.extend([
                {'period': f'{periods[i]} - {periods[i+1]}', 'Variability': val, 'label': labels[col]}
                for val in var_t.flatten()
            ])
            

        
        ## PLOT HISTOGRAMS        
        ax2 = axes2[i]
        ax2.hist(tas_gcm_mean, histtype='stepfilled', color=colors[0], 
                label=labels[0], density=True, range=(270,300), bins=50, alpha = 0.5)
        ax2.hist(tas_gcm_ia_bc_mean, histtype='stepfilled', color=colors[2], 
                label=labels[2], density=True, range=(270,300), bins=50, alpha = 0.5)
        ax2.set_ylim(0, 0.15)
        ax2.set_xlabel(f'{periods[i]} - {periods[i+1]}')
        ax2.legend()
        
    for ax, col in zip(axes1[0], labels):
        ax.set_title(col, fontsize=14)

    fig1.suptitle(f'Temperature changes for {ssp} (Reference : 1980-2014)', fontsize=16)
    fig1.tight_layout(rect=[0.05, 0, 1, 1], pad = 2)
    fig1.savefig(GRAPHS_DIR/f'metrics/{exp}/{exp}_spatial_futur_trend_{ssp}.png')

    fig2.suptitle(f'Monthly temperature ditribution for {ssp} {periods[i]} - {periods[i+1]}')
    fig2.savefig(GRAPHS_DIR/f'metrics/{exp}/{exp}_hist_futur_trend_{ssp}.png')

    df_var_spatial = pd.DataFrame(var_spatial)
    df_var_temporal = pd.DataFrame(var_temporal)

    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 10))
    fig3, axes3 =  plot_variability(fig3, axes3, df_var_spatial, df_var_temporal, periods, labels, colors)
    fig3.suptitle(f'Daily temperature variability for {ssp}', fontsize=16)
    fig3.tight_layout()
    fig3.savefig(GRAPHS_DIR/f'metrics/{exp}/{exp}_variability_futur_trend_{ssp}.png')


