"""
Test script to visualize maps for experiment 3
"""

import sys
sys.path.append('.')

import glob
import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs


from iriscc.settings import GCM_RAW_DIR, PREDICTION_DIR, CONFIG, GRAPHS_DIR, SAFRAN_REFORMAT_DIR, ERA5_DIR
from iriscc.plotutils import plot_map_contour
from iriscc.datautils import standardize_longitudes, crop_domain_from_ds, remove_countries, standardize_dims_and_coords, interpolation_target_grid

date = pd.Timestamp('2014-12-31 00:00:00')
safran = xr.open_mfdataset(np.sort(glob.glob(str(SAFRAN_REFORMAT_DIR/f'tas*reformat.nc'))), combine='by_coords')
safran = safran.sel(time=safran.time.dt.date == date.date()).isel(time=0)
tas_saf = safran.tas.values
tas_saf = remove_countries(tas_saf)
safran.close()


date = pd.Timestamp('2070-01-01 00:00:00')
gcm = xr.open_dataset(glob.glob(str(GCM_RAW_DIR/f'CNRM-CM6-1/tas*ssp585_r1i1p1f2*.nc'))[0])
gcm = gcm.sel(time=gcm.time.dt.date == date.date()).isel(time=0)
gcm = standardize_longitudes(gcm)
gcm = crop_domain_from_ds(gcm, CONFIG['exp3']['domain'])
tas = gcm.tas.values
gcm.close()

date = pd.Timestamp('2014-12-31 00:00:00')
file = glob.glob(str(ERA5_DIR/f'tas/tas*_{date.year}_*'))[0]
ds_era5 = xr.open_dataset(file)
ds_era5 = standardize_dims_and_coords(ds_era5)
ds_era5 = standardize_longitudes(ds_era5)
ds_era5 = ds_era5.reindex(lat=ds_era5.lat[::-1])
ds_era5 = crop_domain_from_ds(ds_era5, CONFIG['exp3']['domain'])
ds_era5 = ds_era5.sel(time=ds_era5.time.dt.date == date.date())
ds_era5 = ds_era5.isel(time=0)
ds_era5_to_gcm = interpolation_target_grid(ds_era5, 
                                                ds_target=gcm, 
                                                method="conservative_normed")
tas_era5 = ds_era5_to_gcm.tas.values

date = pd.Timestamp('2070-01-01 00:00:00')
gcm_unet = xr.open_dataset(glob.glob(str(PREDICTION_DIR/f'tas*ssp585_r1i1p1f2*exp3_unet_all_gcm_bc.nc'))[0])
gcm_unet = gcm_unet.sel(time=gcm_unet.time.dt.date == date.date()).isel(time=0)
tas_unet = gcm_unet.tas.values
gcm_unet.close()

gcm_swinunet = xr.open_dataset(glob.glob(str(PREDICTION_DIR/f'tas*ssp585_r1i1p1f2*exp3_swinunet_all_gcm_bc.nc'))[0])
gcm_swinunet = gcm_swinunet.sel(time=gcm_swinunet.time.dt.date == date.date()).isel(time=0)
tas_swinunet = gcm_swinunet.tas.values
gcm_swinunet.close()



levels = np.linspace(258, 285, 10)
plot_map_contour(tas_saf,
                domain = CONFIG['exp3']['domain_xy'],
                data_projection = CONFIG['exp3']['data_projection'],
                fig_projection = CONFIG['exp3']['fig_projection'],
                title = f'tas SAFRAN 8km 2014-12-31',
                cmap='OrRd',
                var_desc='K',
                levels=levels,
                save_dir=GRAPHS_DIR/'test2.png')

plot_map_contour(tas_era5,
                domain = CONFIG['exp3']['domain'],
                data_projection = ccrs.PlateCarree(),
                fig_projection = CONFIG['exp3']['fig_projection'],
                title = f'tas ERA5 8km 2014-12-31',
                cmap='OrRd',
                var_desc='K',
                levels=levels,
                save_dir=GRAPHS_DIR/'test3.png')

levels = levels = np.linspace(267, 291, 9)
plot_map_contour(tas,
                domain = CONFIG['exp3']['domain'],
                data_projection = ccrs.PlateCarree(),
                fig_projection = CONFIG['exp3']['fig_projection'],
                title = f'tas GCM 1° 2070-01-01',
                cmap='OrRd',
                var_desc='K',
                levels=levels,
                save_dir=GRAPHS_DIR/'test.png')

plot_map_contour(tas_swinunet,
                domain = CONFIG['exp3']['domain_xy'],
                data_projection = CONFIG['exp3']['data_projection'],
                fig_projection = CONFIG['exp3']['fig_projection'],
                title = f'tas SwinUNETR 8km 2070-01-01',
                cmap='OrRd',
                var_desc='K',
                levels=levels,
                save_dir=GRAPHS_DIR/'test0.png')

plot_map_contour(tas_unet,
                domain = CONFIG['exp3']['domain_xy'],
                data_projection = CONFIG['exp3']['data_projection'],
                fig_projection = CONFIG['exp3']['fig_projection'],
                title = f'tas Unet 8km 2070-01-01',
                cmap='OrRd',
                var_desc='K',
                levels=levels,
                save_dir=GRAPHS_DIR/'test1.png')

