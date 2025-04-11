''' Experience 1 pp : SAFRAN interpolated and topography as input and SAFRAN as target'''

import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import glob
from datetime import datetime
import pandas as pd

from iriscc.plotutils import plot_test
from iriscc.datautils import interpolation_target_grid, standardize_longitudes, landseamask_cmip6
from iriscc.settings import (SAFRAN_REFORMAT_DIR, 
                             CMIP6_RAW_DIR,
                             DATES_TEST,
                             OROG_FILE,
                             DATASET_TEST_6MB_ISAFRAN)



def get_cmip6_dataset():
    file = glob.glob(str(CMIP6_RAW_DIR/f'CNRM-CM6-1/tas*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_longitudes(ds)
    ds = ds.sel(lon=slice(-6,12), lat=slice(40.,52.))
    ds = ds.isel(time=0)
    ds = landseamask_cmip6(ds)
    return ds


def target_data(date):
    ''' Returns target data as an array of shape (H, W) '''

    threshold_date = datetime(date.year, 8, 1)
    year = date.year
    if date < threshold_date : 
        year = year-1
    ds = xr.open_dataset(glob.glob(str(SAFRAN_REFORMAT_DIR/f"SAFRAN_{year}080107_{year+1}080106_reformat.nc"))[0])

    if date == datetime(date.year, 8, 1):
        ds_before = xr.open_dataset(glob.glob(str(SAFRAN_REFORMAT_DIR/f"SAFRAN_{year-1}080107_{year}080106_reformat.nc"))[0])
        ds_before = ds_before.isel(time=slice (-7, None))
        ds = ds.isel(time = slice(None, 17))
        ds = ds.merge(ds_before)
    else : 
        ds = ds.sel(time=pd.date_range(start=date.strftime("%Y-%m-%d"), periods = 24, freq='h').to_numpy())

    ds = ds[['tas']]
    ds = ds.reduce(np.nanmean, dim='time')
    return ds


if __name__=='__main__':

    for date in DATES_TEST:
        print(date)

        # HR dataset
        ds_saf = target_data(date) # 2D 
        y = ds_saf.tas.values
        ds_saf["mask"] = xr.where(~np.isnan(ds_saf["tas"]), 1, 0)

        # LR dataset
        ds_cmip6 = get_cmip6_dataset()
        ds_cmip6["mask"] = xr.where(~np.isnan(ds_cmip6["tas"]), 1, 0)

        # Interpolate HR to LR
        ds_saf_to_cmip6 = interpolation_target_grid(ds_saf, ds_target=ds_cmip6, method="conservative_normed")
        ds_saf_to_cmip6["mask"] = xr.where(~np.isnan(ds_saf_to_cmip6["tas"]), 1, 0)
        ds_saf_to_cmip6 = ds_saf_to_cmip6.drop_dims(['x_b', 'y_b'])
              
        # Interpolate LR to HR
        ds_cmip6_to_safran = interpolation_target_grid(ds_saf_to_cmip6, ds_target=ds_saf, method="conservative_normed")

        # Create sample
        tas = ds_cmip6_to_safran['tas'].values
        # Add topography
        ds_z = xr.open_dataset(OROG_FILE)
        z = ds_z['z'].values
        x = np.concatenate([z[np.newaxis, :, :]] + [tas[np.newaxis, :, :]] * 6, axis=0)
        y = np.expand_dims(y, axis=0)
        sample = {'x' : x,
                    'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_TEST_6MB_ISAFRAN/f'sample_{date_str}.npz', **sample)
