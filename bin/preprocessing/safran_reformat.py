"""
Reformat SAFRAN data from raw 1D format to a 2D grid format using cKDTree for interpolation and a reference file.

Localized to follow local repository structure.
date : 16/07/2025
author : Zoé GARCIA
"""

import sys
import os
from pathlib import Path
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
import glob

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parents[2].resolve()
sys.path.append(str(PROJECT_ROOT))

from iriscc.settings import SAFRAN_RAW_DIR, SAFRAN_DIR

def reformat_safran_xy(file):
    # target dataset (grid reference) - MUST be in the local utils directory
    grid_ref = PROJECT_ROOT / 'utils' / 'tasmax_1d_21000101_21001231.nc'
    
    if not grid_ref.exists():
        print(f"CRITICAL ERROR: Grid reference file not found at {grid_ref}. Formatting aborted.")
        return

    ds_grid = xr.open_dataset(grid_ref)
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
    new_ds = ds_grid.drop_vars(['tasmax', 'time'], errors='ignore')

    new_ds = new_ds.expand_dims(dim={'time' : time})
    new_ds['time'] = (['time'], time)
    new_ds['time'].attrs = time_attrs
    new_ds['tas'] = (['time', 'y', 'x'], tas_grid)
    new_ds['tas'].attrs = tas_attrs

    os.makedirs(SAFRAN_DIR, exist_ok=True)
    new_ds.to_netcdf(SAFRAN_DIR/f'{os.path.basename(file)[:-3]}_reformat.nc')

if __name__=='__main__':
    safran_files = glob.glob(str(SAFRAN_RAW_DIR/'SAFRAN*'))
    if not safran_files:
        print(f"No SAFRAN files found in {SAFRAN_RAW_DIR}")
    for file in safran_files:
        print(f"Processing {file}...")
        reformat_safran_xy(file)