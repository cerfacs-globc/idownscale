"""
Useful functions for data processing and reformatting.

date : 21/04/2026
author : Zoé GARCIA / Antigravity (v48)
"""

import sys
import os
sys.path.append('.')

import xarray as xr
import numpy as np
from pathlib import Path
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
                             ERA5_DIR,
                             ERA5_OROG_FILE)


def standardize_dims_and_coords(ds) :
    # v48 Unambiguous Restoration: Establish unique coordinate handles to satisfy cf-xarray lookup.
    dim_mapping = {'x' : ['i', 'ni', 'xh', 'lon', 'nlon', 'longitude'], 
                   'y' : ['j', 'nj', 'yh', 'lat', 'nlat', 'latitude'],
                   'lev' : ['olevel']}
    coord_mapping = {'lon' : ['longitude', 'nav_lon', 'x'],
                     'lat' : ['latitude', 'nav_lat', 'y']}
   
    for standard_name, possible_names in dim_mapping.items() :
        for name in possible_names :
            if name in ds.dims :
                ds = ds.rename({name: standard_name})
                break
   
    for standard_name, possible_names in coord_mapping.items() :
       for name in possible_names :
           if name in ds.coords :
            # Unique Identity Restoration: Each spatial axis has exactly one coordinate name.
            ds = ds.rename({name: standard_name})
            break

    # Identity Restoration (Grace Hopper Compliance):
    if 'lon' in ds.coords:
        ds.lon.attrs['standard_name'] = 'longitude'
        ds.lon.attrs['units'] = 'degrees_east'
        ds.lon.attrs['axis'] = 'X'
    if 'lat' in ds.coords:
        ds.lat.attrs['standard_name'] = 'latitude'
        ds.lat.attrs['units'] = 'degrees_north'
        ds.lat.attrs['axis'] = 'Y'
         
    return ds


def standardize_longitudes(ds) :
    # v47 Deep Index Update: Re-synchronized for the Unambiguous Protocol
    if 'lon' in ds.coords :
        lon = ds.coords['lon']
        ds.coords['lon'] = ((lon+180)%360)-180
        # Sync dimension index with standardized coordinate
        if 'x' in ds.dims:
            ds = ds.assign_coords(x=ds.lon)
      
        if len(ds.lon.shape) == 1 :
            ds = ds.sortby('lon')
        else :
            for dim in ds.lon.dims :
                ds = ds.sortby(dim)
    elif 'x' in ds.coords :
        x = ds.coords['x']
        ds.coords['x'] = ((x+180)%360)-180
        ds = ds.sortby('x')
      
    return ds


def generate_bounds(coord:np.ndarray) -> np.ndarray:
    bounds = np.zeros(len(coord) + 1)
    bounds[1:-1] = 0.5 * (coord[:-1] + coord[1:])
    bounds[0] = coord[0] - (coord[1] - coord[0]) / 2
    bounds[-1] = coord[-1] + (coord[-1] - coord[-2]) / 2
    # v48 Foundation: Maintain float64 for Grace Hopper stability
    return bounds.astype(np.float64)


def add_lon_lat_bounds(ds:xr.Dataset, projection=None, bounds_method="1") -> xr.Dataset:
   # v50 Strict Bounds Purge: Clear existing metadata to avoid alignment conflicts on ARM.
   # We drop both the variables and the dimensions to ensure a clean slate.
   existing_vars = [v for v in ['x_b', 'y_b', 'lon_b', 'lat_b'] if v in ds.variables or v in ds.coords]
   if existing_vars:
       ds = ds.drop_vars(existing_vars, errors='ignore')
   
   if 'x_b' in ds.dims: ds = ds.drop_dims('x_b')
   if 'y_b' in ds.dims: ds = ds.drop_dims('y_b')

   if bounds_method == "1":
      # v48 Unambiguous Access: Use the unique lon/lat coordinates
      x = ds.lon.values
      y = ds.lat.values

      x_b = generate_bounds(x)
      y_b = generate_bounds(y)

      # float64 for ESMF compatibility on ARM
      x_b_2d, y_b_2d = np.meshgrid(x_b, y_b, indexing='xy')

      proj = projection 

      if proj is not None:
         lon_b, lat_b = proj(x_b_2d, y_b_2d, inverse=True)
      else:
         lon_b, lat_b = x_b_2d, y_b_2d

      ds = ds.assign_coords(
         x_b=("x_b", x_b.astype(np.float64)), 
         y_b=("y_b", y_b.astype(np.float64)),
         lon_b=(["y_b", "x_b"], lon_b.astype(np.float64)),
         lat_b=(["y_b", "x_b"], lat_b.astype(np.float64))
      )
   # (Method 2 omitted here as redundant for Era5/GCM regridding)
   return ds


def interpolation_target_grid(ds:xr.Dataset, 
                              ds_target:xr.Dataset, 
                              method:str, 
                              input_projection=None,
                              target_projection=None,
                              bounds_method="1",
                              reuse_weights:bool=False) -> xr.Dataset:
   import xesmf as xe
   for var in ds.data_vars:
         data = ds[var].values
         if data.ndim == 2 or data.ndim == 3:
            ds[var].values = np.asfortranarray(data)
            ds[var].values = np.ascontiguousarray(data)
   if method == 'bilinear':
      regridder = xe.Regridder(ds, ds_target, method, extrap_method="nearest_s2d", reuse_weights=reuse_weights)
   else:
      # v49 Cornerstone Resolution: Force regeneration of bounds to ensure ESMF alignment.
      if 'lon' in ds.coords:
         ds = add_lon_lat_bounds(ds, input_projection, bounds_method)
      if 'lon' in ds_target.coords:
         ds_target = add_lon_lat_bounds(ds_target, target_projection, bounds_method)

      regridder = xe.Regridder(ds, ds_target, method, reuse_weights=reuse_weights)
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
   ds_target = xr.open_dataset(target_file, engine='netcdf4').isel(time=0)
   ds_target = standardize_dims_and_coords(ds_target)
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
   # v47 Direct Coordinate Selection: Immunity to dimension-index ambiguity.
   if domain:
      if 'lon' in ds.coords:
         mask_lon = (ds.lon >= domain[0]) & (ds.lon <= domain[1])
         mask_lat = (ds.lat >= domain[2]) & (ds.lat <= domain[3])
         # Map coordinate masks to correct dimension handles (x/y)
         dim_x = 'x' if 'x' in ds.dims else 'lon'
         dim_y = 'y' if 'y' in ds.dims else 'lat'
         ds = ds.isel({dim_x: mask_lon, dim_y: mask_lat})
      else:
         ds = ds.sel(lon=slice(domain[0], domain[1]), lat=slice(domain[2], domain[3]))
   return ds


def remove_countries(array:np.ndarray) -> np.ndarray:
   ds = xr.open_dataset(COUNTRIES_MASK, engine='netcdf4')
   ds = standardize_dims_and_coords(ds)
   # v48 Unambiguous Sync: Reindex dimension y using coordinate lat
   ds = ds.reindex(y=ds.lat.values[::-1]) 
   ds = crop_domain_from_ds(ds, CONFIG['exp3']['domain'])
   ds = ds.drop_vars('spatial_ref')
   index = ds['index'].values
   countries_to_remove = [41.0, 56.0, 105.0, 112.0, 28.0]
   mask = np.isin(index, countries_to_remove)
   index = np.where(mask, index, np.nan)
   ds['index'].values = index
   ds["mask"] = xr.where(~np.isnan(ds["index"]), 1, 0)
   ds_saf = xr.open_dataset(TARGET_SAFRAN_FILE, engine='netcdf4').isel(time=0)
   ds_saf = standardize_dims_and_coords(ds_saf)
   ds_saf["mask"] = xr.where(~np.isnan(ds_saf["tas"]), 1, 0)
   ds = interpolation_target_grid(ds, ds_saf, method='conservative_normed', target_projection=SAFRAN_PROJ_PYPROJ)
   index = ds['index'].values
   index = xr.where(~np.isnan(index), 1, 0)
   array[index == 1] = np.nan
   return array


def apply_landseamask(ds:xr.Dataset, mask_type:str, variables, domain=None) -> xr.Dataset:
   if mask_type == 'gcm':
      mask = xr.open_dataset(LANDSEAMASK_GCM, engine='netcdf4')
      mask = standardize_longitudes(mask)
      condition = mask['sftlf'].values < 2
   elif mask_type == 'era5':
      mask = xr.open_dataset(LANDSEAMASK_ERA5, engine='netcdf4').isel(time=0)
      mask = standardize_dims_and_coords(mask)
      mask = standardize_longitudes(mask)
      # v48 Unambiguous Sync
      mask = mask.reindex(y=mask.lat.values[::-1])
      condition = mask['lsm'].values < 0.1
   elif mask_type == 'eobs':
      mask = xr.open_dataset(LANDSEAMASK_EOBS, engine='netcdf4')
      mask = standardize_dims_and_coords(mask)
      condition = mask['landseamask'].values == 1.
   else:
      raise ValueError("Invalid mask_type. Choose from 'gcm', 'era5', or 'eobs'.")

   mask = mask.sel(lon=slice(ds['lon'].values.min(), ds['lon'].values.max()),
                   lat=slice(ds['lat'].values.min(), ds['lat'].values.max()))
   for var in variables:
      data = ds[var].values
      data[condition] = np.nan
      ds[var].values = data
      ds["mask"] = xr.where(~np.isnan(ds[var]), 1, 0)
   return ds


class Data(object):
   def __init__(self, domain=None):
      self.domain = domain

   def clean_data(self, data, var, data_type=None):
      if var == 'pr':
         data[data < 0] = 0.
         if data_type == 'gcm' or data_type =='rcm':
            data = data * 3600 * 24
      if var == 'tas':
         if np.nanmean(data) < 100:
            data = data + 273.15
      return data
   
   def get_era5_dataset(self, var:str, date, lapse_rate_correction:bool=False, orog_target_file:str=None, reuse_weights:bool=False):
      import xesmf as xe
      pattern = str(ERA5_DIR / f"{var}_1d" / f"{var}_1d_{date.year}_ERA5.nc")
      files = glob.glob(pattern)
      if not files:
          # Fallback for alternative naming
          pattern = str(ERA5_DIR / f"{var}" / f"{var}_1d_{date.year}_ERA5.nc")
          files = glob.glob(pattern)
      if not files: raise FileNotFoundError(f"Missing ERA5: {pattern}")
      file = files[0]
      ds = xr.open_dataset(file, engine='netcdf4')
      ds = standardize_dims_and_coords(ds)
      ds = standardize_longitudes(ds)

      # v52.2 Solid Dynamic Orientation Protocol (Ascending Europe / Descending France)
      if self.domain and (self.domain[0] == -12.5 or self.domain[2] == 31):
          ds = ds.sortby('lat', ascending=True)
      elif self.domain and (self.domain[0] == -6.0 or self.domain[2] == 38):
          ds = ds.sortby('lat', ascending=False)

      if lapse_rate_correction and var == 'tas' and orog_target_file:
          ds_z = xr.open_dataset(ERA5_OROG_FILE, engine='netcdf4').isel(time=0)
          ds_z = standardize_dims_and_coords(ds_z)
          ds_z = standardize_longitudes(ds_z)
          if self.domain and (self.domain[0] == -12.5 or self.domain[2] == 31):
              ds_z = ds_z.sortby('lat', ascending=True)
          else:
              ds_z = ds_z.sortby('lat', ascending=False)
          h_source = ds_z['z'] / 9.80665
          ds_target_orog = xr.open_dataset(orog_target_file, engine='netcdf4')
          if 'time' in ds_target_orog.dims: ds_target_orog = ds_target_orog.isel(time=0)
          ds_target_orog = standardize_dims_and_coords(ds_target_orog)
          ds_target_orog = standardize_longitudes(ds_target_orog)
          if self.domain and (self.domain[0] == -12.5 or self.domain[2] == 31):
              ds_target_orog = ds_target_orog.sortby('lat', ascending=True)
          else:
              ds_target_orog = ds_target_orog.sortby('lat', ascending=False)
          h_target = ds_target_orog['elevation'] if 'elevation' in ds_target_orog else ds_target_orog['z']
          if 'z' in ds_target_orog: h_target = h_target / 9.80665
          
          w_file_to_era5 = f"weights_target_to_era5_{self.domain[0]}.nc"
          reuse_to_era5 = reuse_weights and os.path.exists(w_file_to_era5)
          regridder_to_era5 = xe.Regridder(h_target, ds, 'bilinear', reuse_weights=reuse_to_era5, filename=w_file_to_era5)
          h_target_coarse = regridder_to_era5(h_target)
          w_file_z_to_era5 = f"weights_source_z_to_era5_{self.domain[0]}.nc"
          reuse_z_to_era5 = reuse_weights and os.path.exists(w_file_z_to_era5)
          regridder_z_to_era5 = xe.Regridder(h_source, ds, 'bilinear', reuse_weights=reuse_z_to_era5, filename=w_file_z_to_era5)
          h_source_coarse = regridder_z_to_era5(h_source)
          delta_h = h_target_coarse - h_source_coarse
          ds[var].values = ds[var].values + (delta_h.values * (-0.0065))
      
      ds = self.crop_time_dim(ds, date)
      ds = crop_domain_from_ds(ds, self.domain)
      ds[var].values = self.clean_data(ds[var].values, var, data_type='era5')
      return ds

   def get_gcm_dataset(self, var:str, date, ssp:str=None, lapse_rate_correction:bool=False, orog_target_file:str=None, reuse_weights:bool=False):
      import xesmf as xe
      if date is None or date < pd.Timestamp('2015-01-01'):
         file = glob.glob(str(GCM_RAW_DIR/f'CNRM-CM6-1/{var}*historical*r1i1p1f2*'))[0]
      else:
         file = glob.glob(str(GCM_RAW_DIR/f'CNRM-CM6-1/{var}*{ssp}*'))[0]
      ds = xr.open_dataset(file, engine='netcdf4')
      ds = standardize_dims_and_coords(ds)
      ds = standardize_longitudes(ds)

      # v52.2 Solid Dynamic Orientation Protocol
      if self.domain and (self.domain[0] == -6.0 or self.domain[2] == 38): # France
          ds = ds.sortby('lat', ascending=False)
      elif self.domain and (self.domain[0] == -12.5 or self.domain[2] == 31): # Europe
          ds = ds.sortby('lat', ascending=True)

      # v86.7: Full Implementation of missing Lapse Rate Correction for GCM Gaussian grid
      if lapse_rate_correction and var == 'tas' and orog_target_file:
          gcm_orog_path = '/scratch/globc/page/idownscale_rerun/rawdata/gcm/orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc'
          ds_z = xr.open_dataset(gcm_orog_path, engine='netcdf4')
          if 'time' in ds_z.dims: ds_z = ds_z.isel(time=0)
          ds_z = standardize_dims_and_coords(ds_z)
          ds_z = standardize_longitudes(ds_z)
          if self.domain and (self.domain[0] == -12.5 or self.domain[2] == 31):
              ds_z = ds_z.sortby('lat', ascending=True)
          else:
              ds_z = ds_z.sortby('lat', ascending=False)
          h_source = ds_z['orog']
          
          ds_target_orog = xr.open_dataset(orog_target_file, engine='netcdf4')
          if 'time' in ds_target_orog.dims: ds_target_orog = ds_target_orog.isel(time=0)
          ds_target_orog = standardize_dims_and_coords(ds_target_orog)
          ds_target_orog = standardize_longitudes(ds_target_orog)
          if self.domain and (self.domain[0] == -12.5 or self.domain[2] == 31):
              ds_target_orog = ds_target_orog.sortby('lat', ascending=True)
          else:
              ds_target_orog = ds_target_orog.sortby('lat', ascending=False)
          h_target = ds_target_orog['elevation'] if 'elevation' in ds_target_orog else ds_target_orog['z']
          if 'z' in ds_target_orog: h_target = h_target / 9.80665

          w_file_to_gcm = f"weights_target_to_gcm_{self.domain[0]}.nc"
          reuse_to_gcm = reuse_weights and os.path.exists(w_file_to_gcm)
          regridder_to_gcm = xe.Regridder(h_target, ds, 'bilinear', reuse_weights=reuse_to_gcm, filename=w_file_to_gcm)
          h_target_coarse = regridder_to_gcm(h_target)
          w_file_z_to_gcm = f"weights_source_z_to_gcm_{self.domain[0]}.nc"
          reuse_z_to_gcm = reuse_weights and os.path.exists(w_file_z_to_gcm)
          regridder_z_to_gcm = xe.Regridder(h_source, ds, 'bilinear', reuse_weights=reuse_z_to_gcm, filename=w_file_z_to_gcm)
          h_source_coarse = regridder_z_to_gcm(h_source)
          
          delta_h = h_target_coarse - h_source_coarse
          ds[var].values = ds[var].values + (delta_h.values * (-0.0065))
      
      ds = self.crop_time_dim(ds, date)
      ds = crop_domain_from_ds(ds, self.domain)
      ds[var].values = self.clean_data(ds[var].values, var, data_type='gcm')
      return ds

   
   def get_rcm_dataset(self, var:str, date, ssp:str=None, lapse_rate_correction:bool=False, orog_target_file:str=None):
      import xesmf as xe
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
               ds = xr.open_dataset(file, engine='netcdf4')
               ds = self.crop_time_dim(ds, date)
      if 'x' not in ds.dims:
         ds = ds.assign_coords(x = (['x'], xref))
         ds = ds.assign_coords(y = (['y'], yref))
      ds['x'] = ds['x'].values * 1000
      ds['y'] = ds['y'].values * 1000
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
      ds = xr.open_dataset(file, engine='netcdf4')
      ds = self.crop_time_dim(ds, date)
      ds = standardize_dims_and_coords(ds)
      # v48 Unambiguous Sync
      ds = ds.reindex(y=ds.lat.values[::-1])
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
   match var:
      case 'tas': return 'K'
      case 'pr': return 'mm/day'
      case 'sfcWind': return 'm/s'
      case 'psl': return 'Pa'
      case _: raise ValueError(f"Unknown variable: {var}")