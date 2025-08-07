"""
Useful functions for data processing and reformatting.

date : 16/07/2025
author : Zoé GARCIA
"""

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
                             SAFRAN_REFORMAT_DIR,
                             EOBS_RAW_DIR,
                             GCM_RAW_DIR,
                             COUNTRIES_MASK,
                             LANDSEAMASK_GCM,
                             LANDSEAMASK_ERA5,
                             LANDSEAMASK_EOBS,
                             SAFRAN_PROJ_PYPROJ,
                             CONFIG,
                             RCM_RAW_DIR,
                             ERA5_DIR)




def standardize_dims_and_coords(ds) :
    # Camille Le Gloannec script
    # GCM models have inconsistent names of dimensions and coordinates, 
    # this function fix that at the dataset level by naming dimensions (x,y) and coordinates (lon,lat).

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
    # GCM models have inconsistent longitude conventions, this function fix 
    # that at the dataset level by setting the convention to -180° - 180°.

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


def add_lon_lat_bounds(ds:xr.Dataset, projection=None, bounds_method="1") -> xr.Dataset:
   """
   Adds longitude and latitude bounds to the dataset based on the coordinates of the cells using exact data projection
   """
   if bounds_method == "1":
      x = ds['x'].values
      y = ds['y'].values

      x_b = generate_bounds(x)
      y_b = generate_bounds(y)

      x_b_2d, y_b_2d = np.meshgrid(x_b, y_b)

      proj = projection 

      lon_b, lat_b = proj(x_b_2d, y_b_2d, inverse=True)

      ds = ds.assign_coords(
         x_b=("x_b", x_b), 
         y_b=("y_b", y_b),
         lon_b=(["y_b", "x_b"], lon_b),
         lat_b=(["y_b", "x_b"], lat_b)
      )

   if bounds_method == "2":
      """
      Adds longitude and latitude bounds to the dataset based on the coordinates of the cells.
      """
      lon = ds["lon"]  
      lat = ds["lat"]
      print(lon.shape, lat.shape)

      lon_b = 0.25 * (lon[:-1, :-1] + lon[1:, :-1] + lon[:-1, 1:] + lon[1:, 1:])
      lat_b = 0.25 * (lat[:-1, :-1] + lat[1:, :-1] + lat[:-1, 1:] + lat[1:, 1:])
      print(lon_b.shape, lat_b.shape)

      nx_b = lon.shape[0] + 1
      ny_b = lon.shape[1] + 1
      print(ny_b, nx_b)

      lon_b_full = np.full((nx_b, ny_b), np.nan)
      lat_b_full = np.full((nx_b, ny_b), np.nan)
      print(lon_b_full.shape, lat_b_full.shape)
      lon_b_full[1:-2, 1:-2] = lon_b
      lat_b_full[1:-2, 1:-2] = lat_b

      lon_b_full[0, 1:-1] = 2 * lon[0, :-1] - lon[1, :-1]
      lon_b_full[-1, 1:-1] = 2 * lon[-1, :-1] - lon[-2, :-1]
      lon_b_full[1:-1, 0] = 2 * lon[:-1, 0] - lon[:-1, 1]
      lon_b_full[1:-1, -1] = 2 * lon[:-1, -1] - lon[:-1, -2]

      lon_b_full[0, 0]     = 2 * lon[0, 0]     - lon[1, 1]
      lon_b_full[0, -1]    = 2 * lon[0, -1]    - lon[1, -2]
      lon_b_full[-1, 0]    = 2 * lon[-1, 0]    - lon[-2, 1]
      lon_b_full[-1, -1]   = 2 * lon[-1, -1]   - lon[-2, -2]

      lat_b_full[0, 1:-1] = 2 * lat[0, :-1] - lat[1, :-1]
      lat_b_full[-1, 1:-1] = 2 * lat[-1, :-1] - lat[-2, :-1]
      lat_b_full[1:-1, 0] = 2 * lat[:-1, 0] - lat[:-1, 1]
      lat_b_full[1:-1, -1] = 2 * lat[:-1, -1] - lat[:-1, -2]

      lat_b_full[0, 0]     = 2 * lat[0, 0]     - lat[1, 1]
      lat_b_full[0, -1]    = 2 * lat[0, -1]    - lat[1, -2]
      lat_b_full[-1, 0]    = 2 * lat[-1, 0]    - lat[-2, 1]
      lat_b_full[-1, -1]   = 2 * lat[-1, -1]   - lat[-2, -2]

      ds = ds.assign_coords(
         lon_b=(["y_b", "x_b"], lon_b_full),
         lat_b=(["y_b", "x_b"], lat_b_full)
      )

   return ds


def interpolation_target_grid(ds:xr.Dataset, 
                              ds_target:xr.Dataset, 
                              method:str, 
                              input_projection=None,
                              target_projection=None,
                              bounds_method="1") -> xr.Dataset:
   """
   Interpolates the input dataset to match the target grid and domain.
   """
   

   for var in ds.data_vars:
         data = ds[var].values
         if data.ndim == 2 or data.ndim == 3:
            ds[var].values = np.asfortranarray(data)
            ds[var].values = np.ascontiguousarray(data)
   if method == 'bilinear':
      regridder = xe.Regridder(ds, ds_target, method, extrap_method="nearest_s2d")
   else:
      if 'x' in ds.dims :
         if 'x_b' not in ds.coords:
            ds = add_lon_lat_bounds(ds, input_projection, bounds_method)
      if 'x' in ds_target.dims :
         if 'x_b' not in ds_target.coords:
            ds_target = add_lon_lat_bounds(ds_target, target_projection, bounds_method)

      regridder = xe.Regridder(ds, ds_target, method)
   ds_out = regridder(ds)
   return ds_out


def reformat_as_target(ds:xr.Dataset, 
                       target_file, 
                       method:str, 
                       domain:tuple, 
                       mask:bool=False,
                       crop_input:bool=False, 
                       crop_target:bool=False, 
                       input_projection=None,
                       target_projection=None) -> xr.Dataset:
   """
   Reformats the input dataset to match the target grid and domain.
   """
   
   ds_target = xr.open_dataset(target_file).isel(time=0)
   ds_target = standardize_longitudes(ds_target)
   if crop_input:
      ds = crop_domain_from_ds(ds, domain)
   if crop_target:
      ds_target = crop_domain_from_ds(ds_target, domain)
   if mask :
      if 'mask' not in list(ds_target.keys()):
         ds_target["mask"] = xr.where(~np.isnan(ds_target["tas"]), 1, 0)
   ds = interpolation_target_grid(ds, 
                                  ds_target, 
                                  method, 
                                  input_projection,
                                  target_projection)
   return ds


def crop_domain_from_ds(ds:xr.Dataset, domain:tuple) -> xr.Dataset:
   """
   Crops the input dataset to a specified geographical domain based on latitude and longitude coordinates.
   """
   if domain:
      if 'x' in ds.dims:
         ds = ds.sel(x=slice(domain[0], domain[1]), y=slice(domain[2], domain[3]))
      else:
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
   ds = crop_domain_from_ds(ds, CONFIG['exp3']['domain'])  # Crop to France domain
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
   ds = interpolation_target_grid(ds, ds_saf, method='conservative_normed', target_projection=SAFRAN_PROJ_PYPROJ)
   index = ds['index'].values

   # Apply the mask to the input array
   index = xr.where(~np.isnan(index), 1, 0)
   array[index == 1] = np.nan
   return array


def apply_landseamask(ds:xr.Dataset, mask_type:str, variables, domain=None) -> xr.Dataset:
   """
   Apply a land-sea mask to the dataset based on the specified mask type.

   Returns:
   xarray.Dataset: The dataset with the land-sea mask applied.
   """

   if mask_type == 'gcm':
      mask = xr.open_dataset(LANDSEAMASK_GCM)
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
      raise ValueError("Invalid mask_type. Choose from 'gcm', 'era5', or 'eobs'.")

   # Align mask with the dataset's spatial domain
   mask = mask.sel(lon=slice(ds['lon'].values.min(), ds['lon'].values.max()),
                   lat=slice(ds['lat'].values.min(), ds['lat'].values.max()))
   for var in variables:
      print(var)
      data = ds[var].values
      data[condition] = np.nan  # Apply the mask condition
      ds[var].values = data
      ds["mask"] = xr.where(~np.isnan(ds[var]), 1, 0)
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

class Data(object):
   def __init__(self, domain=None):
      self.domain = domain

   def clean_data(self, data, var, data_type=None):
      if var == 'pr':
         data[data < 0] = 0.
         if data_type == 'gcm' or data_type =='rcm': # kg/m2/s to mm/day
               data = data * 3600 * 24
      if var == 'tas':
         if np.nanmean(data) < 100: # celsius to kelvin
               data = data + 273.15
      return data
   
   def get_era5_dataset(self, var:str, date):
      file = glob.glob(str(ERA5_DIR/f'{var}/{var}*_{date.year}_*'))[0]
      ds = xr.open_dataset(file)
      ds = standardize_dims_and_coords(ds)
      ds = standardize_longitudes(ds)
      ds = ds.reindex(lat=ds.lat[::-1])
      ds = crop_domain_from_ds(ds, self.domain)
      ds = self.crop_time_dim(ds, date)
      ds[var].values = self.clean_data(ds[var].values, var, data_type='era5')
      return ds
   
   def get_gcm_dataset(self, var:str, date, ssp:str=None):
      if date is None or date < pd.Timestamp('2015-01-01'):
         file = glob.glob(str(GCM_RAW_DIR/f'CNRM-CM6-1/{var}*historical*r1i1p1f2*'))[0]
      else:
         file = glob.glob(str(GCM_RAW_DIR/f'CNRM-CM6-1/{var}*{ssp}*'))[0]
      ds = xr.open_dataset(file)
      ds = standardize_longitudes(ds)
      ds = self.crop_time_dim(ds, date)
      ds = crop_domain_from_ds(ds, self.domain)
      ds[var].values = self.clean_data(ds[var].values, var, data_type='gcm')
      return ds
   
   def get_rcm_dataset(self, var:str, date, ssp:str=None):
      if date is None:
         file = glob.glob(str(RCM_RAW_DIR / f'ALADIN/{var}*ssp585*r1i1p1f2*'))[0]
         ds = xr.open_dataset(file).isel(time=0)
      else :
         if date < pd.Timestamp('2015-01-01'):
               file_for_xy = glob.glob(str(RCM_RAW_DIR / f'ALADIN/{var}*ssp585*r1i1p1f2*'))[0]
               ds_for_xy = xr.open_dataset(file_for_xy).isel(time=0)
               xref = ds_for_xy['x'].values
               yref = ds_for_xy['y'].values
               ds_for_xy.close()
               files = np.sort(glob.glob(str(RCM_RAW_DIR / f'ALADIN/{var}*historical*r1i1p1f2*')))
         else :
               files = np.sort(glob.glob(str(RCM_RAW_DIR / f'ALADIN/{var}*{ssp}*r1i1p1f2*')))
         for file in files:
               if int(file.split('_')[-1][:4]) <= date.year <= int(file.split('_')[-1][9:13]):
                  ds = xr.open_dataset(file)
                  ds = self.crop_time_dim(ds, date)
      if 'x' not in ds.coords:
         ds = ds.assign_coords(x = (['x'], xref))
         ds = ds.assign_coords(y = (['y'], yref))

      x = ds['x'].values * 1000 # in meter to match Lambert Conformal projection
      y = ds['y'].values * 1000
      ds['x'] = x
      ds['y'] = y
      ds[var].values = self.clean_data(ds[var].values, var, data_type='rcm')
      return ds
   
   def get_safran_dataset(self, var:str, date):
      ds = xr.open_dataset(glob.glob(str(SAFRAN_REFORMAT_DIR/f"{var}*{date.year}_reformat.nc"))[0])
      ds = self.crop_time_dim(ds, date)
      ds[var].values = self.clean_data(ds[var].values, var, data_type='safran')
      ds[var].values = remove_countries(ds[var].values)
      return ds

   def get_eobs_dataset(self, var:str, date):
      file = glob.glob(str(EOBS_RAW_DIR/f'{var}*'))[0]
      ds = xr.open_dataset(file)
      ds = self.crop_time_dim(ds, date)
      ds = standardize_dims_and_coords(ds)
      ds = apply_landseamask(ds, 'eobs', variables=[var])
      ds = crop_domain_from_ds(ds, self.domain)
      ds[var].values = self.clean_data(ds[var].values, var, data_type='eobs')
      return ds
   
   def crop_time_dim(self, ds, date=None):
      if date is not None:
         ds = ds.sel(time=ds.time.dt.date == date.date())
         ds = ds.isel(time=0)
      return ds
   
   def get_target_dataset(self, target:str, var:str='tas', date=None) -> xr.Dataset:
      if target == 'safran':
         ds = self.get_safran_dataset(var, date)
      elif target == 'eobs':
         ds = self.get_eobs_dataset(var, date)
      return ds
   

def return_unit(var:str):
   """
   Returns the unit of measurement for a given variable.

   Parameters:
      var (str): The variable name for which the unit is requested.
               Accepted values are 'tas', 'pr', 'sfcWind', and 'psl'.

   Returns:
      str: The unit of measurement corresponding to the variable.

   Raises:
      ValueError: If the variable name is not recognized.
   """
   match var:
      case 'tas':
         return 'K'
      case 'pr':
         return 'mm/day'
      case 'sfcWind':
         return 'm/s'
      case 'psl':
         return 'Pa'
      case _:
         raise ValueError(f"Unknown variable: {var}")