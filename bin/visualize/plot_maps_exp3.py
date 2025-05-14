import sys
sys.path.append('.')

import glob
import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs


from iriscc.settings import CMIP6_RAW_DIR, PREDICTION_DIR, CONFIG, GRAPHS_DIR, SAFRAN_REFORMAT_DIR
from iriscc.plotutils import plot_map_contour
from iriscc.datautils import standardize_longitudes, crop_domain_from_ds, remove_countries

date = pd.Timestamp('2014-12-31 00:00:00')

gcm = xr.open_dataset(glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*historical_r1i1p1f2*.nc'))[0])
gcm = gcm.sel(time=gcm.time.dt.date == date.date()).isel(time=0)
gcm = standardize_longitudes(gcm)
gcm = crop_domain_from_ds(gcm, CONFIG['safran']['domain']['france'])
tas = gcm.tas.values
gcm.close()

gcm_unet = xr.open_dataset(glob.glob(str(PREDICTION_DIR/f'tas*historical_r1i1p1f2*exp3_unet_all_cmip6_bc.nc'))[0])
gcm_unet = gcm_unet.sel(time=gcm_unet.time.dt.date == date.date()).isel(time=0)
tas_unet = gcm_unet.tas.values
gcm_unet.close()

gcm_swinunet = xr.open_dataset(glob.glob(str(PREDICTION_DIR/f'tas*historical_r1i1p1f2*exp3_swinunet_all_cmip6_bc.nc'))[0])
gcm_swinunet = gcm_swinunet.sel(time=gcm_swinunet.time.dt.date == date.date()).isel(time=0)
tas_swinunet = gcm_swinunet.tas.values
gcm_swinunet.close()

safran = xr.open_mfdataset(np.sort(glob.glob(str(SAFRAN_REFORMAT_DIR/f'tas*reformat.nc'))), combine='by_coords')
safran = safran.sel(time=safran.time.dt.date == date.date()).isel(time=0)
tas_saf = safran.tas.values
tas_saf = remove_countries(tas_saf)
safran.close()

levels = np.linspace(258, 285, 10)

plot_map_contour(tas,
                domain = CONFIG['safran']['domain']['france'],
                data_projection = ccrs.PlateCarree(),
                fig_projection = CONFIG['safran']['fig_projection']['france_xy'],
                title = f'tas GCM 1° 2014-12-31',
                cmap='OrRd',
                var_desc='K',
                levels=levels,
                save_dir=GRAPHS_DIR/'test.png')

plot_map_contour(tas_swinunet,
                domain = CONFIG['safran']['domain']['france_xy'],
                data_projection = CONFIG['safran']['data_projection'],
                fig_projection = CONFIG['safran']['fig_projection']['france_xy'],
                title = f'tas SwinUNETR 8km 2014-12-31',
                cmap='OrRd',
                var_desc='K',
                levels=levels,
                save_dir=GRAPHS_DIR/'test0.png')

plot_map_contour(tas_unet,
                domain = CONFIG['safran']['domain']['france_xy'],
                data_projection = CONFIG['safran']['data_projection'],
                fig_projection = CONFIG['safran']['fig_projection']['france_xy'],
                title = f'tas Unet 8km 2014-12-31',
                cmap='OrRd',
                var_desc='K',
                levels=levels,
                save_dir=GRAPHS_DIR/'test1.png')

plot_map_contour(tas_saf,
                domain = CONFIG['safran']['domain']['france_xy'],
                data_projection = CONFIG['safran']['data_projection'],
                fig_projection = CONFIG['safran']['fig_projection']['france_xy'],
                title = f'tas SAFRAN 8km 2014-12-31',
                cmap='OrRd',
                var_desc='K',
                levels=levels,
                save_dir=GRAPHS_DIR/'test2.png')

