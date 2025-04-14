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


from iriscc.settings import CMIP6_RAW_DIR, PREDICTION_DIR, CONFIG, GRAPHS_DIR
from iriscc.plotutils import plot_test
from iriscc.datautils import standardize_longitudes, crop_domain_from_ds, landseamask_cmip6

exp = str(sys.argv[1])
model = str(sys.argv[2])
ssp = str(sys.argv[3])

## Réference 1980-2014
gcm_ref = xr.open_dataset(glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*historical_r1i1p1f2*.nc'))[0])
gcm_ref = standardize_longitudes(gcm_ref)
gcm_ref = crop_domain_from_ds(gcm_ref, CONFIG['safran']['domain']['france'])

gcm_ref = gcm_ref.sel(time=slice('1980-01-01', '2014-12-31'))
tas_ref = np.mean(gcm_ref.tas.values) # mean spatial reference tas 
gcm_ref.close()

periods = ['2015', '2040', '2070', '2100']   



output_projection = CONFIG['safran']['fig_projection']['france_xy']
inputs_projections = [ccrs.PlateCarree(), 
                      CONFIG['safran']['data_projection'], 
                      CONFIG['safran']['data_projection']]
domain = [CONFIG['safran']['domain']['france'],
            CONFIG['safran']['domain']['france_xy'],
            CONFIG['safran']['domain']['france_xy']]

fig, axes = plt.subplots(
    3, 3,
    figsize=(15, 10),
    subplot_kw={'projection': output_projection}
)
 
for i in range(len(periods)-1):
    gcm = xr.open_dataset(glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*{ssp}_r1i1p1f2*.nc'))[0])
    gcm = standardize_longitudes(gcm)
    gcm = crop_domain_from_ds(gcm, CONFIG['safran']['domain']['france'])
    gcm = gcm.sel(time=slice(periods[i], periods[i+1]))
    tas_gcm = np.nanmean(gcm.tas.values, axis=0) - tas_ref
    gcm.close()
    ax = axes[i, 0]
    gcm_ia = xr.open_dataset(glob.glob(str(PREDICTION_DIR/f'tas*{ssp}_r1i1p1f2*{exp}_{model}_cmip6.nc'))[0])
    gcm_ia = gcm_ia.sel(time=slice(periods[i], periods[i+1]))
    tas_gcm_ia = np.nanmean(gcm_ia.tas.values, axis=0) - tas_ref
    gcm_ia.close()

    gcm_ia_bc = xr.open_dataset(glob.glob(str(PREDICTION_DIR/f'tas*{ssp}_r1i1p1f2*{exp}_{model}_cmip6_bc.nc'))[0])
    gcm_ia_bc = gcm_ia_bc.sel(time=slice(periods[i], periods[i+1]))
    tas_gcm_ia_bc = np.nanmean(gcm_ia_bc.tas.values, axis=0) - tas_ref
    gcm_ia_bc.close()
    title = ['raw CNRM-CM6-1', 'IA CNRM-CM6-1', 'IA CNRM-CM6-1 bc']
    l = [tas_gcm, tas_gcm_ia, tas_gcm_ia_bc]
    for col in range(3):
        ax = axes[i, col]
        data_proj = inputs_projections[col]
        img = ax.imshow(
            l[col],
            extent=domain[col],
            transform=data_proj,
            origin='lower',
            cmap='bwr',
            vmin=-10,
            vmax=10
        )
        if col == 0:
            ax.set_ylabel(f'{periods[i]} - {periods[i+1]}')

        ax.set_extent([-5., 11., 41., 51.], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
        ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, edgecolor='gray', zorder=10)
        ax.set_title(title[col], fontsize=12)
        cbar = plt.colorbar(img, ax=ax, pad=0.05, shrink=0.8)
        cbar.set_label(label='K', size=12)

plt.tight_layout()
plt.savefig(GRAPHS_DIR/f'metrics/{exp}/{exp}_spatial_futur_trend_{ssp}_{model}.png')
