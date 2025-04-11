import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import glob

from iriscc.plotutils import plot_test
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

def generate_bounds(coord):
      bounds = np.zeros(len(coord) + 1)
      bounds[1:-1] = 0.5 * (coord[:-1] + coord[1:])  # Milieux entre chaque point
      bounds[0] = coord[0] - (coord[1] - coord[0]) / 2  # Première limite extrapolée
      bounds[-1] = coord[-1] + (coord[-1] - coord[-2]) / 2  # Dernière limite extrapolée
      return bounds.astype(np.int32)

def add_lon_lat_bounds(ds):
   ''' Irregular grid '''
   ''' Generate boundaries coordonates from cells center for the consevative interpolation method '''

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



def interpolation_target_grid(ds, ds_target, method):

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


def reformat_as_target(ds, target_file, method, domain, mask:bool=False, crop_target:bool=False):
   ''' Returns Input dataset interpolated at target target grid '''
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

def crop_domain_from_ds(ds, domain):
   ds = ds.sel(lon=slice(domain[0], domain[1]), lat=slice(domain[2], domain[3]))
   return ds

def remove_countries(array):
   # the array must have SAFRAN shape
   ds = xr.open_dataset(COUNTRIES_MASK)
   ds = ds.reindex(lat=ds.lat[::-1])
   ds = crop_domain_from_ds(ds, CONFIG['safran']['domain']['france'])
   ds = ds.drop_vars('spatial_ref')
   index = ds['index'].values
   pays = [41.0, 56., 105., 112., 28.] # Suisse, Allemagne, Autriche, Italie
   mask = np.isin(index, pays)
   index = np.where(mask, index, np.nan)

   ds['index'].values = index
   ds["mask"] = xr.where(~np.isnan(ds["index"]), 1, 0)
   ds_saf = xr.open_dataset(TARGET_SAFRAN_FILE).isel(time=0)
   ds_saf["mask"] = xr.where(~np.isnan(ds_saf["tas"]), 1, 0)
   ds = interpolation_target_grid(ds, ds_saf, method='conservative_normed')
   index = ds['index'].values

   index = xr.where(~np.isnan(index), 1, 0)
   array[index == 1] = np.nan
   return array

def landseamask_cmip6(ds):
   tas = ds['tas'].values
   mask = xr.open_dataset(LANDSEAMASK_CMIP6)
   mask = standardize_longitudes(mask)
   mask = mask.sel(lon=slice(ds['lon'].values.min(), ds['lon'].values.max()),
                   lat=slice(ds['lat'].values.min(), ds['lat'].values.max()))
   condition = mask['sftlf'].values < 2 # 2
   tas[condition] = np.nan
   ds["mask"] = xr.where(~np.isnan(ds["tas"]), 1, 0)
   ds['tas'].values = tas
   return ds

def landseamask_era5(ds):
   tas = ds['tas'].values
   mask = xr.open_dataset(LANDSEAMASK_ERA5).isel(time=0)
   mask = standardize_dims_and_coords(mask)
   mask = standardize_longitudes(mask)
   mask = mask.reindex(lat=mask.lat[::-1])
   mask = mask.sel(lon=slice(ds['lon'].values.min(), ds['lon'].values.max()),
                   lat=slice(ds['lat'].values.min(), ds['lat'].values.max()))
   condition = mask['lsm'].values < 0.1
   tas[condition] = np.nan
   ds['tas'].values = tas
   ds["mask"] = xr.where(~np.isnan(ds["tas"]), 1, 0)
   return ds

def landseamask_eobs(ds):
   tas = ds['tas'].values
   mask = xr.open_dataset(LANDSEAMASK_EOBS)
   mask = standardize_dims_and_coords(mask)
   mask = mask.sel(lon=slice(ds['lon'].values.min(), ds['lon'].values.max()),
                   lat=slice(ds['lat'].values.min(), ds['lat'].values.max()))
   condition = mask['landseamask'].values == 1.
   tas[condition] = np.nan
   ds['tas'].values = tas
   ds["mask"] = xr.where(~np.isnan(ds["tas"]), 1, 0)
   return ds

def crop_domain_from_array(array, sample_dir, domain):
   coords_file = glob.glob(str(sample_dir/'coordinates.npz'))[0]
   coordinates = dict(np.load(coords_file, allow_pickle=True))
   lon = coordinates['lon']
   lat = coordinates['lat']
   lon_indices = np.where((lon >= domain[0]) & (lon <= domain[1]))[0]
   lat_indices = np.where((lat >= domain[2]) & (lat <= domain[3]))[0]
   array = array[np.ix_(lat_indices, lon_indices)]
   return array
