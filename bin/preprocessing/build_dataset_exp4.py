''' Experience 2 : ERA5 and topography as input and SAFRAN as target'''

import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import glob

from iriscc.plotutils import plot_test
from iriscc.datautils import (standardize_dims_and_coords, 
                              standardize_longitudes, 
                              interpolation_target_grid, 
                              reformat_as_target,
                              crop_domain_from_ds,
                              landseamask_eobs)
from iriscc.settings import (DATES,
                             DATES_TEST,
                             ERA5_DIR,
                             EOBS_RAW_DIR,
                             DATASET_EXP4_30Y_DIR,
                             CONFIG,
                             TARGET,
                             OROG_FILE,
                             CMIP6_RAW_DIR,
                             TARGET_EOBS_FILE,
                             INPUTS)

def get_era5_dataset(date, domain):
    file = glob.glob(str(ERA5_DIR/f'tas*_{date.year}_*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_dims_and_coords(ds)
    ds = standardize_longitudes(ds)
    ds = ds.reindex(lat=ds.lat[::-1])
    ds = crop_domain_from_ds(ds, domain)
    ds = ds.sel(time=ds.time.dt.date == date.date())
    ds = ds.isel(time=0)
    return ds

def get_cmip6_dataset(domain):
    file = glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_longitudes(ds)
    ds = crop_domain_from_ds(ds, domain)
    ds = ds.isel(time=0)
    return ds

def input_data(date, domain):
    ''' Returns inputs data as an array of shape (C, H, W) '''
    x = []

    # Commune variables
    ds = xr.open_dataset(OROG_FILE) # Already interpolated to target grids
    ds = crop_domain_from_ds(standardize_dims_and_coords(ds), domain)
    x.append(ds['elevation'].values)

    for var in INPUTS:
        ds_era5 = get_era5_dataset(date, domain)
        ds_cmip6 = get_cmip6_dataset(domain)
        ds_era5_to_cmip6 = interpolation_target_grid(ds_era5, 
                                                     ds_target=ds_cmip6, 
                                                     method="conservative_normed")
        ds = reformat_as_target(ds_era5_to_cmip6, 
                                target_file=TARGET_EOBS_FILE, 
                                method='conservative_normed',
                                domain=domain,
                                crop_target=True)
        x.append(ds[var].values)
    x = np.concatenate([x[0][np.newaxis, :, :]] + [x[1][np.newaxis, :, :]] * 1, axis=0)

    return x


def target_data(date, domain):
    ''' Returns target data as an array of shape (H, W) '''
    file = glob.glob(str(EOBS_RAW_DIR/f'tas*'))[0]
    ds = xr.open_dataset(file)
    ds = ds.sel(time=ds.time.dt.date == date.date()).isel(time=0)
    ds = standardize_dims_and_coords(ds)
    ds = apply_landseamask(ds, 'eobs')
    ds = crop_domain_from_ds(ds, domain)
    lon = ds['lon'].values
    lat = ds['lat'].values
    tas = ds[TARGET].values
    y = np.expand_dims(tas, axis=0)
    if np.nanmean(y) < 100: # if celsus
        y = y + 273.15
    return y, lon, lat


if __name__=='__main__':
    domain = CONFIG['eobs']['domain']['europe']
    for i, date in enumerate(DATES):
        print(date)

        x = input_data(date, domain)
        if i == 0:
            y, lon, lat = target_data(date, domain)
            coordinates = {'lon': lon,
                    'lat': lat}
            np.savez(DATASET_EXP4_30Y_DIR/f'coordinates.npz', **coordinates)
        else:
            y, _, _ = target_data(date, domain)
        sample = {'x' : x,
                  'y' : y}
         
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_EXP4_30Y_DIR/f'sample_{date_str}.npz', **sample)

    
