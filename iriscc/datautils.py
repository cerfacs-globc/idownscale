"""
Useful functions for data processing and reformatting.

date : 21/04/2026
author : Zoé GARCIA / Antigravity (v48)
"""

import xarray as xr
import numpy as np
import os
from pathlib import Path
import glob
from datetime import datetime
import pandas as pd

from iriscc.settings import (TARGET_SAFRAN_FILE,
                             ALADIN_PROJ_PYPROJ,
                             SAFRAN_REFORMAT_DIR,
                             EOBS_RAW_DIR,
                             GCM_RAW_DIR,
                             GCM_OROG_FILE,
                             COUNTRIES_MASK,
                             LANDSEAMASK_GCM,
                             LANDSEAMASK_ERA5,
                             LANDSEAMASK_EOBS,
                             SAFRAN_PROJ_PYPROJ,
                             CONFIG,
                             RCM_RAW_DIR,
                             ERA5_DIR,
                             ERA5_OROG_FILE,
                             REGRID_WEIGHTS_DIR,
                             SOURCE_CATALOG)


def _grid_signature(ds: xr.Dataset) -> str:
   lon = np.asarray(ds.lon.values)
   lat = np.asarray(ds.lat.values)
   return (
      f"{lon.shape[0]}x{lat.shape[0]}_"
      f"{float(np.nanmin(lon)):.6f}_{float(np.nanmax(lon)):.6f}_"
      f"{float(np.nanmin(lat)):.6f}_{float(np.nanmax(lat)):.6f}"
   )

def standardize_era5_geometry(ds):
    # v86.67 Native-First: Minimal Renaming for ESMF recognition
    rename_dict = {}
    if 'longitude' in ds.coords: rename_dict['longitude'] = 'lon'
    if 'latitude' in ds.coords: rename_dict['latitude'] = 'lat'
    if rename_dict: ds = ds.rename(rename_dict)
    return ds

def standardize_gcm_geometry(ds):
    # v86.67 Native-First: Minimal Renaming for ESMF recognition
    rename_dict = {}
    if 'longitude' in ds.coords: rename_dict['longitude'] = 'lon'
    if 'latitude' in ds.coords: rename_dict['latitude'] = 'lat'
    if rename_dict: ds = ds.rename(rename_dict)
    return ds

def standardize_eobs_geometry(ds):
    # v86.67 Post-Regrid Protocol: x/y Compatibility Bridge
    rename_dict = {}
    if 'longitude' in ds.coords: rename_dict['longitude'] = 'lon'
    if 'latitude' in ds.coords: rename_dict['latitude'] = 'lat'
    if rename_dict: ds = ds.rename(rename_dict)
    
    # x/y coordinate aliasing for pipeline compatibility
    if 'lon' in ds.coords: ds = ds.assign_coords(x=ds.lon)
    if 'lat' in ds.coords: ds = ds.assign_coords(y=ds.lat)
    return ds

def ARCHIVAL_standardize_dims_and_coords(ds) :
    # v48 DEPRECATED: Standardize via Named Plugins instead
    dim_mapping = {'x' : ['i', 'ni', 'xh', 'nlon'], 
                   'y' : ['j', 'nj', 'yh', 'nlat']}
    for standard_name, possible_names in dim_mapping.items() :
        for name in possible_names :
            if name in ds.dims :
                ds = ds.rename({name: standard_name})
                break
    return ds



def standardize_longitudes(ds) :
    # v65faa6 Archival Sync: Mandatory -180..180 Shift and Monotonic Sort
    if 'lon' in ds.coords:
        lon = np.asarray(ds.coords['lon'].values)
        ds = ds.assign_coords(lon=("lon", np.array(((lon + 180) % 360) - 180, copy=True)))
        if len(ds.lon.shape) == 1:
            ds = ds.sortby('lon')
    elif 'x' in ds.coords :
        x = np.asarray(ds.x.values)
        ds = ds.assign_coords(x=("x", np.array((((x + 180) % 360) - 180), copy=True)))
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
      if ds.lon.ndim == 1 and ds.lat.ndim == 1:
         x = ds.lon.values
         y = ds.lat.values
      elif projection is not None and 'x' in ds.coords and 'y' in ds.coords and ds.x.ndim == 1 and ds.y.ndim == 1:
         x = ds.x.values
         y = ds.y.values
      else:
         raise ValueError(
            "Cannot generate conservative bounds for this grid: expected 1D lon/lat "
            "or 1D x/y with a projection."
         )

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


def attach_native_cf_bounds(ds: xr.Dataset) -> xr.Dataset:
   """Expose native curvilinear bounds through CF metadata when available."""
   if 'bounds_lon' not in ds.variables or 'bounds_lat' not in ds.variables:
      return ds
   if 'lon' not in ds.coords or 'lat' not in ds.coords:
      return ds

   lon_bounds = ds['bounds_lon']
   lat_bounds = ds['bounds_lat']
   if 'time' in lon_bounds.dims:
      lon_bounds = lon_bounds.isel(time=0, drop=True)
   if 'time' in lat_bounds.dims:
      lat_bounds = lat_bounds.isel(time=0, drop=True)
   if 'nvertex' not in lon_bounds.dims or 'nvertex' not in lat_bounds.dims:
      return ds

   spatial_dims = tuple(dim for dim in lon_bounds.dims if dim != 'nvertex')
   if len(spatial_dims) != 2:
      return ds

   ds = ds.copy()
   ds['lon_bounds'] = (('nvertex',) + spatial_dims, np.moveaxis(lon_bounds.values, -1, 0))
   ds['lat_bounds'] = (('nvertex',) + spatial_dims, np.moveaxis(lat_bounds.values, -1, 0))
   ds['lon'].attrs['bounds'] = 'lon_bounds'
   ds['lat'].attrs['bounds'] = 'lat_bounds'
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
   w_file = REGRID_WEIGHTS_DIR / (
      f"weights_{method}_{_grid_signature(ds)}_to_{_grid_signature(ds_target)}.nc"
   )
   reuse = reuse_weights and os.path.exists(w_file)
   if method == 'bilinear':
      regridder = xe.Regridder(ds, ds_target, method, extrap_method="nearest_s2d", reuse_weights=reuse, filename=str(w_file))
   else:
      ds = attach_native_cf_bounds(ds)
      ds_target = attach_native_cf_bounds(ds_target)
      # v49 Cornerstone Resolution: Force regeneration of bounds to ensure ESMF alignment.
      if 'lon' in ds.coords and 'lon_b' not in ds and 'lon_bounds' not in ds.variables:
         ds = add_lon_lat_bounds(ds, input_projection, bounds_method)
      if 'lon' in ds_target.coords and 'lon_b' not in ds_target and 'lon_bounds' not in ds_target.variables:
         ds_target = add_lon_lat_bounds(ds_target, target_projection, bounds_method)

      regridder = xe.Regridder(ds, ds_target, method, reuse_weights=reuse, filename=str(w_file))
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
                       target_projection=None,
                       reuse_weights:bool=False) -> xr.Dataset:
    if isinstance(target_file, (str, Path)):
       ds_target = xr.open_dataset(str(target_file), engine='netcdf4')
    else:
       ds_target = target_file
    
    if 'time' in ds_target.dims:
       ds_target = ds_target.isel(time=0, drop=True)
    
    # v48 Plugin Target: Standardized according to dataset type (EOBS/SAFRAN)
    if 'safran' in str(target_file).lower():
       ds_target = ARCHIVAL_standardize_dims_and_coords(ds_target)
    else:
       ds_target = standardize_eobs_geometry(ds_target)
       
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
                                   target_projection,
                                   reuse_weights=reuse_weights)
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
   ds = standardize_eobs_geometry(ds)
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
   ds_saf = ARCHIVAL_standardize_dims_and_coords(ds_saf)
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
      mask = standardize_era5_geometry(mask)
      mask = standardize_longitudes(mask)
      condition = mask['lsm'].values < 0.1
   elif mask_type == 'eobs':
      mask = xr.open_dataset(LANDSEAMASK_EOBS, engine='netcdf4')
      mask = standardize_eobs_geometry(mask)
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

   def get_source_spec(self, source_name: str) -> dict:
      if source_name not in SOURCE_CATALOG:
         raise KeyError(f"Unknown source '{source_name}'. Add it to SOURCE_CATALOG in iriscc/settings.py.")
      return SOURCE_CATALOG[source_name]

   def _standardize_source_geometry(self, ds: xr.Dataset, geometry: str) -> xr.Dataset:
      if geometry == 'era5':
         ds = standardize_era5_geometry(ds)
         ds = standardize_longitudes(ds)
         if 'lat' in ds.coords:
            ds = ds.reindex(lat=ds.lat[::-1])
         return ds
      if geometry == 'gcm':
         ds = standardize_gcm_geometry(ds)
         ds = standardize_longitudes(ds)
         return ds
      if geometry == 'eobs':
         ds = standardize_eobs_geometry(ds)
         return ds
      if geometry == 'safran':
         return ds
      if geometry == 'rcm':
         if 'x' in ds.dims and 'x' not in ds.coords:
            ds = ds.assign_coords(x=('x', np.arange(ds.sizes['x'], dtype=np.float64) * 1000.0))
         if 'y' in ds.dims and 'y' not in ds.coords:
            ds = ds.assign_coords(y=('y', np.arange(ds.sizes['y'], dtype=np.float64) * 1000.0))
         return ds
      return ds

   def _resolve_source_file(self, source_name: str, var: str, date=None, ssp: str | None = None) -> str:
      spec = self.get_source_spec(source_name)
      root = Path(spec['root'])
      candidates: list[str] = []

      if 'yearly_patterns' in spec:
         if date is None:
            raise ValueError(f"Source '{source_name}' requires a date for yearly file lookup.")
         for pattern in spec['yearly_patterns']:
            candidates.extend(glob.glob(str(root / pattern.format(var=var, year=date.year, ssp=ssp or ""))))
      elif 'yearly_pattern' in spec:
         if date is None:
            raise ValueError(f"Source '{source_name}' requires a date for yearly file lookup.")
         candidates.extend(glob.glob(str(root / spec['yearly_pattern'].format(var=var, year=date.year, ssp=ssp or ""))))
      else:
         if date is None or date < pd.Timestamp('2015-01-01'):
            pattern = spec.get('historical_pattern') or spec.get('scenario_pattern') or spec.get('glob_pattern')
            if pattern is not None:
               candidates.extend(glob.glob(str(root / pattern.format(var=var, ssp=ssp or ""))))
         else:
            pattern = spec.get('scenario_pattern') or spec.get('historical_pattern') or spec.get('glob_pattern')
            if pattern is not None:
               candidates.extend(glob.glob(str(root / pattern.format(var=var, ssp=ssp or ""))))

      if not candidates:
         raise FileNotFoundError(f"Missing source file for source='{source_name}', var='{var}', date='{date}', ssp='{ssp}'")
      candidates = list(np.sort(np.array(candidates)))
      if date is not None and spec.get('geometry') == 'rcm':
         for candidate in candidates:
            tail = Path(candidate).name.split('_')[-1]
            if len(tail) >= 17 and tail[:8].isdigit() and tail[9:17].isdigit():
               start_year = int(tail[:4])
               end_year = int(tail[9:13])
               if start_year <= date.year <= end_year:
                  return candidate
      return candidates[0]

   def _open_source_dataset(self, source_name: str, var: str, date=None, ssp: str | None = None) -> xr.Dataset:
      spec = self.get_source_spec(source_name)
      file = self._resolve_source_file(source_name, var, date=date, ssp=ssp)
      ds = xr.open_dataset(file, engine='netcdf4')
      ds = self._standardize_source_geometry(ds, spec.get('geometry', 'none'))
      return ds

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
      return self.get_reanalysis_dataset(
         'era5',
         var,
         date,
         lapse_rate_correction=lapse_rate_correction,
         orog_target_file=orog_target_file,
         reuse_weights=reuse_weights,
      )

   def get_reanalysis_dataset(
      self,
      source_name: str,
      var: str,
      date,
      lapse_rate_correction: bool = False,
      orog_target_file: str = None,
      reuse_weights: bool = False,
   ):
      import xesmf as xe
      spec = self.get_source_spec(source_name)
      ds = self._open_source_dataset(source_name, var, date=date)

      if lapse_rate_correction and var == 'tas' and orog_target_file:
          # Native-First Induction: Regrid Raw Orog to Target before Standardizing
          source_orog_file = spec.get('orography_file', ERA5_OROG_FILE)
          ds_z = xr.open_dataset(source_orog_file, engine='netcdf4').isel(time=0)
          # Minimal Input Sync
          if 'longitude' in ds_z.coords: ds_z = ds_z.rename({'longitude':'lon', 'latitude':'lat'})
          ds_z = standardize_longitudes(ds_z)
          h_source = (ds_z['z'] / 9.80665).fillna(0)
          ds_target_orog = xr.open_dataset(orog_target_file, engine='netcdf4')
          if 'time' in ds_target_orog.dims: ds_target_orog = ds_target_orog.isel(time=0)
          h_target = (ds_target_orog['elevation'] if 'elevation' in ds_target_orog else ds_target_orog['z']).fillna(0)
          if 'z' in ds_target_orog: h_target = h_target / 9.80665
          
          w_file_to_era5 = REGRID_WEIGHTS_DIR / f"weights_target_to_era5_{self.domain[0]}.nc"
          reuse_to_era5 = reuse_weights and os.path.exists(w_file_to_era5)
          regridder_to_era5 = xe.Regridder(h_target, ds, 'bilinear', reuse_weights=reuse_to_era5, filename=str(w_file_to_era5))
          h_target_coarse = regridder_to_era5(h_target)
          w_file_z_to_era5 = REGRID_WEIGHTS_DIR / f"weights_source_z_to_era5_{self.domain[0]}.nc"
          reuse_z_to_era5 = reuse_weights and os.path.exists(w_file_z_to_era5)
          regridder_z_to_era5 = xe.Regridder(h_source, ds, 'bilinear', reuse_weights=reuse_z_to_era5, filename=str(w_file_z_to_era5))
          h_source_coarse = regridder_z_to_era5(h_source)
          
          # RAW Inductive Subtraction (No alias noise)
          delta_h = h_target_coarse - h_source_coarse
          
          # Post-Subtraction Standardization
          ds = standardize_eobs_geometry(ds)
          delta_h = standardize_eobs_geometry(delta_h)
          
          ds[var].values = ds[var].values + (delta_h.values * (-0.0065))
      
      # v48 Reference: Crop AFTER correction
      ds = self.crop_time_dim(ds, date)
      ds = crop_domain_from_ds(ds, self.domain)
      ds[var].values = self.clean_data(ds[var].values, var, data_type=spec.get('data_type', 'era5'))
      return ds

   def get_gcm_dataset(self, var:str, date, ssp:str=None, lapse_rate_correction:bool=False, orog_target_file:str=None, reuse_weights:bool=False):
      return self.get_model_dataset(
         'gcm_cnrm_cm6_1',
         var,
         date,
         ssp=ssp,
         lapse_rate_correction=lapse_rate_correction,
         orog_target_file=orog_target_file,
         reuse_weights=reuse_weights,
      )

   def get_model_dataset(
      self,
      source_name: str,
      var: str,
      date,
      ssp: str = None,
      lapse_rate_correction: bool = False,
      orog_target_file: str = None,
      reuse_weights: bool = False,
   ):
      import xesmf as xe
      spec = self.get_source_spec(source_name)
      if spec.get('geometry') == 'rcm':
         return self.get_rcm_dataset_from_source(
            source_name,
            var,
            date,
            ssp=ssp,
            lapse_rate_correction=lapse_rate_correction,
            orog_target_file=orog_target_file,
         )
      ds = self._open_source_dataset(source_name, var, date=date, ssp=ssp)

      if lapse_rate_correction and var == 'tas' and orog_target_file:
          source_orog_file = spec.get('orography_file', GCM_OROG_FILE)
          ds_z = xr.open_dataset(source_orog_file, engine='netcdf4')
          if 'time' in ds_z.dims: ds_z = ds_z.isel(time=0)
          # Minimal Input Sync
          if 'longitude' in ds_z.coords: ds_z = ds_z.rename({'longitude':'lon', 'latitude':'lat'})
          ds_z = standardize_longitudes(ds_z)
          h_source = ds_z['orog'].fillna(0)
          
          ds_target_orog = xr.open_dataset(orog_target_file, engine='netcdf4')
          if 'time' in ds_target_orog.dims: ds_target_orog = ds_target_orog.isel(time=0)
          h_target = (ds_target_orog['elevation'] if 'elevation' in ds_target_orog else ds_target_orog['z']).fillna(0)
          if 'z' in ds_target_orog: h_target = h_target / 9.80665

          w_file_to_gcm = f"weights_target_to_gcm_{self.domain[0]}.nc"
          reuse_to_gcm = reuse_weights and os.path.exists(w_file_to_gcm)
          regridder_to_gcm = xe.Regridder(h_target, ds, 'bilinear', reuse_weights=reuse_to_gcm, filename=w_file_to_gcm)
          h_target_coarse = regridder_to_gcm(h_target)
          w_file_z_to_gcm = f"weights_source_z_to_gcm_{self.domain[0]}.nc"
          reuse_z_to_gcm = reuse_weights and os.path.exists(w_file_z_to_gcm)
          regridder_z_to_gcm = xe.Regridder(h_source, ds, 'bilinear', reuse_weights=reuse_z_to_gcm, filename=w_file_z_to_gcm)
          h_source_coarse = regridder_z_to_gcm(h_source)
          
          # RAW Inductive Subtraction (No alias noise)
          delta_h = h_target_coarse - h_source_coarse
          
          # Post-Subtraction Standardization
          ds = standardize_eobs_geometry(ds)
          delta_h = standardize_eobs_geometry(delta_h)
          
          ds[var].values = ds[var].values + (delta_h.values * (-0.0065))
      
      # v48 Reference: Crop AFTER correction
      ds = self.crop_time_dim(ds, date)
      ds = crop_domain_from_ds(ds, self.domain)
      ds[var].values = self.clean_data(ds[var].values, var, data_type=spec.get('data_type', 'gcm'))
      return ds

   
   def get_rcm_dataset(self, var:str, date, ssp:str=None, lapse_rate_correction:bool=False, orog_target_file:str=None):
      return self.get_rcm_dataset_from_source('rcm_aladin', var, date, ssp=ssp, lapse_rate_correction=lapse_rate_correction, orog_target_file=orog_target_file)

   def get_rcm_dataset_from_source(self, source_name: str, var: str, date, ssp: str = None, lapse_rate_correction: bool = False, orog_target_file: str = None):
      spec = self.get_source_spec(source_name)
      import xesmf as xe
      root = Path(spec['root'])
      if not root.exists():
         raise FileNotFoundError(
            f"Missing root directory for model source '{source_name}': {root}. "
            f"Set the corresponding source override env var or populate the raw-data layout."
         )

      def resolve_matches(*pattern_keys: str) -> list[str]:
         matches: list[str] = []
         for key in pattern_keys:
            pattern = spec.get(key)
            if pattern is None:
               continue
            matches.extend(glob.glob(str(root / pattern.format(var=var, ssp=ssp or "ssp585"))))
         return list(np.sort(np.array(matches))) if matches else []

      def require_matches(context: str, *pattern_keys: str) -> list[str]:
         matches = resolve_matches(*pattern_keys)
         if not matches:
            requested = [key for key in pattern_keys if spec.get(key) is not None]
            raise FileNotFoundError(
               f"Missing RCM source files for source='{source_name}', var='{var}', date='{date}', "
               f"ssp='{ssp}', root='{root}', searched_patterns={requested}"
            )
         return matches

      if date is None:
         file = require_matches("date=None", 'scenario_pattern', 'historical_pattern')[0]
         ds = xr.open_dataset(file).isel(time=0)
      else :
         if date < pd.Timestamp('2015-01-01'):
            # Some RCM layouts do not ship scenario files next to historical ones,
            # so bootstrap coordinates from whichever native file exists first.
            file_for_xy = require_matches("coordinate bootstrap", 'scenario_pattern', 'historical_pattern')[0]
            ds_for_xy = xr.open_dataset(file_for_xy).isel(time=0)
            xref = ds_for_xy['x'].values if 'x' in ds_for_xy.coords else ds_for_xy['x'].values
            yref = ds_for_xy['y'].values if 'y' in ds_for_xy.coords else ds_for_xy['y'].values
            ds_for_xy.close()
            files = require_matches("historical lookup", 'historical_pattern')
         else :
            files = require_matches("scenario lookup", 'scenario_pattern')
         ds = None
         for file in files:
            if int(file.split('_')[-1][:4]) <= date.year <= int(file.split('_')[-1][9:13]):
               ds = xr.open_dataset(file, engine='netcdf4')
               ds = self.crop_time_dim(ds, date)
               break
         if ds is None:
            raise FileNotFoundError(
               f"No RCM file span matched source='{source_name}', var='{var}', date='{date}', "
               f"ssp='{ssp}' under root='{root}'."
            )
      if 'x' not in ds.dims:
         ds = ds.assign_coords(x = (['x'], xref))
         ds = ds.assign_coords(y = (['y'], yref))
      ds['x'] = ds['x'].values * 1000
      ds['y'] = ds['y'].values * 1000
      ds[var].values = self.clean_data(ds[var].values, var, data_type=spec.get('data_type', 'rcm'))
      return ds
   
   def get_safran_dataset(self, var:str, date):
      ds = self._open_source_dataset('safran', var, date=date)
      ds = self.crop_time_dim(ds, date)
      ds[var].values = self.clean_data(ds[var].values, var, data_type='safran')
      ds[var].values = remove_countries(ds[var].values)
      return ds

   def get_eobs_dataset(self, var:str, date):
      return self.get_observation_dataset('eobs', var, date)

   def get_observation_dataset(self, source_name: str, var: str, date):
      spec = self.get_source_spec(source_name)
      ds = self._open_source_dataset(source_name, var, date=date)
      ds = self.crop_time_dim(ds, date)
      if spec.get('geometry') == 'eobs':
         ds = standardize_eobs_geometry(ds)
         ds = ds.reindex(y=ds.lat.values[::-1])
      mask_type = spec.get('mask_type')
      if mask_type:
         ds = apply_landseamask(ds, mask_type, variables=[var])
      ds = crop_domain_from_ds(ds, self.domain)
      ds[var].values = self.clean_data(ds[var].values, var, data_type=spec.get('data_type', source_name))
      return ds
   
   def crop_time_dim(self, ds, date=None):
       if date is not None:
          # v86.74 Parity Protocol: Aggregate all hours for the requested date into a Daily Mean.
          y_match = ds.time.dt.year.values == date.year
          m_match = ds.time.dt.month.values == date.month
          d_match = ds.time.dt.day.values == date.day
          
          indices = np.where(y_match & m_match & d_match)[0]
          
          if len(indices) > 0:
              # Perform Daily Mean to match archival EOBS-to-ERA5 temporal definition
              ds = ds.isel(time=indices).mean('time')
          else:
              # Final Fallback: Nearest selection via string
              print(f"WARNING: Primitive match failed for {date.date()}. Using nearest.")
              ds = ds.sel(time=date.strftime('%Y-%m-%d'), method='nearest')
              if 'time' in ds.dims and ds.time.size > 1: ds = ds.mean('time')
       return ds
   
   def get_target_dataset(self, target:str, var:str='tas', date=None, source_name: str | None = None) -> xr.Dataset:
      if source_name is not None:
         ds = self.get_observation_dataset(source_name, var, date)
      elif target == 'safran':
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
