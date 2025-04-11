import sys
sys.path.append('.')

import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import glob
import os

from iriscc.settings import SAFRAN_RAW_DIR, SAFRAN_DIR

def reformat_safran_xy(file):

    # target dataset
    ds_grid = xr.open_dataset('/gpfs-calypso/scratch/globc/garcia/utils/tasmax_1d_21000101_21001231.nc')
    lon_grid = ds_grid['lon'].values
    lat_grid = ds_grid['lat'].values
    dimx = np.shape(lon_grid)[1]
    dimy = np.shape(lon_grid)[0]

    # interpolated to safran grid (x,y)
    ds = xr.open_dataset(file)
    lon = ds['LON'].values
    lat = ds['LAT'].values
    tas = ds['Tair']
    tas_attrs = ds['Tair'].attrs
    time = ds['time']
    time_attrs = ds['time'].attrs

    grid_points = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))
    data_points = np.column_stack((lon, lat))

    tree = cKDTree(grid_points)
    _, indices = tree.query(data_points)

    tas_grid = np.nan * np.ones((len(time.values), dimy, dimx))

    id_y, id_x = np.unravel_index(indices, np.shape(lon_grid))
    tas_grid[:, id_y, id_x] = tas.values


    # create a new conforme dataset 
    new_ds = ds_grid.drop_vars(['tasmax', 'time'])

    new_ds.expand_dims(dim={'time' : time})
    new_ds['time'] = (['time'], time)
    new_ds['time'].attrs = time_attrs
    new_ds['tas'] = (['time', 'y', 'x'], tas_grid)
    new_ds['tas'].attrs = tas_attrs

    #new_ds.to_netcdf(SAFRAN_DIR/f'{os.path.basename(file)[:-3]}_reformat.nc')

if __name__=='__main__':
    safran_files = glob.glob(str(SAFRAN_RAW_DIR/'SAFRAN*'))
    for file in safran_files:
        print(file)
        reformat_safran_xy(file)

    