import sys
sys.path.append('.')

import xarray as xr
import numpy as np
from pathlib import Path
import xesmf as xe
import glob
from datetime import datetime
import pandas as pd

from iriscc.settings import (TARGET_SAFRAN_FILE,
                             TARGET_EOBS_FILE,
                             CMIP6_RAW_DIR,
                             COUNTRIES_MASK,
                             LANDSEAMASK_CMIP6,
                             LANDSEAMASK_ERA5,
                             LANDSEAMASK_EOBS,
                             TARGET_SIZE,
                             SAFRAN_PROJ_PYPROJ,
                             CONFIG)




def standardize_dims_and_coords(ds) :
   # Camille Le Gloannec script
   # CMIP6 models have inconsistent names of dimensions and coordinates, this function fix that at the dataset level by naming dimensions (x,y) and coordinates (lon,lat).

   dim_mapping = {'x' : ['i', 'ni', 'xh', 'lon', 'nlon'], 
         'y' : ['j', 'nj', 'yh', 'lat', 'nlat'],
         'lev' : ['olevel']}
   coord_mapping = {'lon' : ['longitude', 'nav_lon'],
         'lat' : ['latitude', 'nav_lat']}
   
   for standard_name, possible_names in dim_mapping.items() :
      for name in possible_names :
         if name in ds.dims :
            ds = ds.rename({name: standard_name})
            break
   
   for standard_name, possible_names in coord_mapping.items() :
      for name in possible_names :
         if name in ds.coords :
            ds = ds.rename({name: standard_name})
            break
         
   return ds


def standardize_longitudes(ds) :
   # Camille Le Gloannec script
   # CMIP6 models have inconsistent longitude conventions, this function fix that at the dataset level by setting the convention to -180° - 180°.

   if 'lon' in ds.coords :
      lon = ds.coords['lon']
      ds.coords['lon'] = ((lon+180)%360)-180
      
      if len(ds.lon.shape) == 1 :
         ds = ds.sortby('lon')
      else :
         for dim in ds.lon.dims :
            ds = ds.sortby(dim)
         
   else :
      x = ds.coords['x']
      ds.coords['x'] = ((x+180)%360)-180
      ds = ds.sortby(ds.x)
      
   return ds


def generate_bounds(coord:np.ndarray) -> np.ndarray:
   """
   Generates bounds for a given coordinate array.
   """
   bounds = np.zeros(len(coord) + 1)
   bounds[1:-1] = 0.5 * (coord[:-1] + coord[1:])  # Milieux entre chaque point
   bounds[0] = coord[0] - (coord[1] - coord[0]) / 2  # Première limite extrapolée
   bounds[-1] = coord[-1] + (coord[-1] - coord[-2]) / 2  # Dernière limite extrapolée
   return bounds.astype(np.int32)


def add_lon_lat_bounds(ds:xr.Dataset) -> xr.Dataset:
   """
   Adds longitude and latitude bounds to the dataset based on the coordinates of the cells.
   Useful for SAFRAN-like datasets.
   """

   x = ds['x'].values
   y = ds['y'].values

   x_b = generate_bounds(x)
   y_b = generate_bounds(y)

   x_b_2d, y_b_2d = np.meshgrid(x_b, y_b)

   projection = SAFRAN_PROJ_PYPROJ
   lon_b, lat_b = projection(x_b_2d, y_b_2d, inverse=True)

   ds = ds.assign_coords(
      x_b=("x_b", x_b), 
      y_b=("y_b", y_b),
      lon_b=(["y_b", "x_b"], lon_b),
      lat_b=(["y_b", "x_b"], lat_b)
   )
   
   return ds


def interpolation_target_grid(ds:xr.Dataset, ds_target:xr.Dataset, method:str) -> xr.Dataset:
   """
   Interpolates the input dataset to match the target grid and domain.
   """

   if 'x' in ds.coords :
      if 'x_b' not in ds.coords:
         ds = add_lon_lat_bounds(ds)
   if 'x' in ds_target.coords :
      if 'x_b' not in ds_target.coords:
         ds_target = add_lon_lat_bounds(ds_target)


   for i, coord in enumerate(['lat','lon']):
      if len(ds[coord].dims) == 1:
         if len(ds[coord].values) > TARGET_SIZE[i]: # if resolution is finer than target's
               new_coord = np.linspace(ds[coord].values.min(), ds[coord].values.max(), TARGET_SIZE[i])
               ds = ds.interp({coord:new_coord})
   for var in ds.data_vars:
      ds[var].values = np.asfortranarray(ds[var].values)
      ds[var].values = np.ascontiguousarray(ds[var].values)
   if method == 'bilinear':
      regridder = xe.Regridder(ds, ds_target, method, extrap_method="nearest_s2d")
   else:
      regridder = xe.Regridder(ds, ds_target, method)
   ds_out = regridder(ds)
   return ds_out


def reformat_as_target(ds:xr.Dataset, target_file, method:str, domain:tuple, 
                       mask:bool=False, crop_target:bool=False) -> xr.Dataset:
   """
   Reformats the input dataset to match the target grid and domain.
   """
   ds = crop_domain_from_ds(ds, domain)
   ds_target = xr.open_dataset(target_file).isel(time=0)
   ds_target = standardize_dims_and_coords(ds_target)
   ds_target = standardize_longitudes(ds_target)
   if crop_target:
      ds_target = crop_domain_from_ds(ds_target, domain)
   if mask :
      if 'mask' not in list(ds_target.keys()):
         ds_target["mask"] = xr.where(~np.isnan(ds_target["tas"]), 1, 0)
   ds = interpolation_target_grid(ds, ds_target, method)
   return ds


def crop_domain_from_ds(ds:xr.Dataset, domain:tuple) -> xr.Dataset:
   """
   Crops the input dataset to a specified geographical domain based on latitude and longitude coordinates.
   """
   ds = ds.sel(lon=slice(domain[0], domain[1]), lat=slice(domain[2], domain[3]))
   return ds


def remove_countries(array:np.ndarray) -> np.ndarray:
   """
   Removes specific countries from the input SAFRAN-like array.

   Args:
      array (np.ndarray): The input 2D array to be modified.

   Returns:
      np.ndarray: The modified array with specific countries removed.
   """
   # Load the countries mask dataset
   ds = xr.open_dataset(COUNTRIES_MASK)
   ds = ds.reindex(lat=ds.lat[::-1])  # Reverse latitude order if necessary
   ds = crop_domain_from_ds(ds, CONFIG['safran']['domain']['france'])  # Crop to France domain
   ds = ds.drop_vars('spatial_ref')  # Drop unnecessary variable
   index = ds['index'].values

   # Define country codes to remove (e.g., Switzerland, Germany, Austria, Italy)
   countries_to_remove = [41.0, 56.0, 105.0, 112.0, 28.0]
   mask = np.isin(index, countries_to_remove)
   index = np.where(mask, index, np.nan)

   # Update the dataset with the modified index
   ds['index'].values = index
   ds["mask"] = xr.where(~np.isnan(ds["index"]), 1, 0)

   # Interpolate the mask to match the SAFRAN grid
   ds_saf = xr.open_dataset(TARGET_SAFRAN_FILE).isel(time=0)
   ds_saf["mask"] = xr.where(~np.isnan(ds_saf["tas"]), 1, 0)
   ds = interpolation_target_grid(ds, ds_saf, method='conservative_normed')
   index = ds['index'].values

   # Apply the mask to the input array
   index = xr.where(~np.isnan(index), 1, 0)
   array[index == 1] = np.nan
   return array


def apply_landseamask(ds:xr.Dataset, mask_type:str) -> xr.Dataset:
   """
   Apply a land-sea mask to the dataset based on the specified mask type.

   Returns:
   xarray.Dataset: The dataset with the land-sea mask applied.
   """
   tas = ds['tas'].values

   if mask_type == 'cmip6':
      mask = xr.open_dataset(LANDSEAMASK_CMIP6)
      mask = standardize_longitudes(mask)
      condition = mask['sftlf'].values < 2  # Land fraction less than 2%
   elif mask_type == 'era5':
      mask = xr.open_dataset(LANDSEAMASK_ERA5).isel(time=0)
      mask = standardize_dims_and_coords(mask)
      mask = standardize_longitudes(mask)
      mask = mask.reindex(lat=mask.lat[::-1])
      condition = mask['lsm'].values < 0.1  # Land-sea mask threshold
   elif mask_type == 'eobs':
      mask = xr.open_dataset(LANDSEAMASK_EOBS)
      mask = standardize_dims_and_coords(mask)
      condition = mask['landseamask'].values == 1.  # Land-sea mask value
   else:
      raise ValueError("Invalid mask_type. Choose from 'cmip6', 'era5', or 'eobs'.")

   # Align mask with the dataset's spatial domain
   mask = mask.sel(lon=slice(ds['lon'].values.min(), ds['lon'].values.max()),
                   lat=slice(ds['lat'].values.min(), ds['lat'].values.max()))

   # Apply the mask
   tas[condition] = np.nan
   ds['tas'].values = tas
   ds["mask"] = xr.where(~np.isnan(ds["tas"]), 1, 0)
   return ds


def crop_domain_from_array(array: np.ndarray, sample_dir: Path, domain: tuple) -> np.ndarray:
   """
   Crops a 2D array to a specified geographical domain based on latitude and longitude coordinates.

   Args:
      array (np.ndarray): The input 2D array to be cropped.
      sample_dir (Path): The directory containing the 'coordinates.npz' file with latitude and longitude data.
      domain (tuple): A tuple specifying the cropping domain in the format (min_lon, max_lon, min_lat, max_lat).

   Returns:
      np.ndarray: The cropped 2D array restricted to the specified domain.
   """
   coords_file = glob.glob(str(sample_dir/'coordinates.npz'))[0]
   coordinates = dict(np.load(coords_file, allow_pickle=True))
   lon = coordinates['lon']
   lat = coordinates['lat']
   lon_indices = np.where((lon >= domain[0]) & (lon <= domain[1]))[0]
   lat_indices = np.where((lat >= domain[2]) & (lat <= domain[3]))[0]
   array = array[np.ix_(lat_indices, lon_indices)]
   return array

def datetime_period_to_string(dates):
   """
   Converts a list of dates to a string representation of the period.

   Args:
      dates (list): A list of dates in various formats (e.g., np.datetime64, pandas.Timestamp, datetime.datetime, or string).

   Returns:
      str: A string representing the period in the format 'DD/MM/YYYY - DD/MM/YYYY'.
   """

   def to_datetime(date):
      if isinstance(date, (np.datetime64, pd.Timestamp)):
         return pd.to_datetime(date).to_pydatetime()
      elif isinstance(date, str):
         return datetime.strptime(date, '%Y-%m-%d')
      elif isinstance(date, datetime):
         return date
      else:
         raise ValueError(f"Unsupported date format: {type(date)}")

   startdate = to_datetime(dates[0]).strftime('%d/%m/%Y')
   enddate = to_datetime(dates[-1]).strftime('%d/%m/%Y')
   period = f'{startdate}-{enddate}'
   return period