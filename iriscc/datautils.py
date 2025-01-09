import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import numpy.ma as ma

from iriscc.settings import (TARGET_GRID_FILE,
                             TARGET_SIZE,
                             LONMIN,
                             LONMAX,
                             LATMIN,
                             LATMAX,
                             TARGET_PROJ_PYPROJ)
from iriscc.plotutils import plot_image, plot_contour



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



def add_lon_lat_bounds(ds):
   ''' Irregular grid '''
   ''' Generate boundaries coordonates from cells center for the consevative interpolation method '''

   def generate_bounds(coord):
      bounds = np.zeros(len(coord) + 1)
      bounds[1:-1] = 0.5 * (coord[:-1] + coord[1:])  # Milieux entre chaque point
      bounds[0] = coord[0] - (coord[1] - coord[0]) / 2  # Première limite extrapolée
      bounds[-1] = coord[-1] + (coord[-1] - coord[-2]) / 2  # Dernière limite extrapolée
      return bounds.astype(np.int32)

   x = ds['x'].values
   y = ds['y'].values

   x_b = generate_bounds(x)
   y_b = generate_bounds(y)

   x_b_2d, y_b_2d = np.meshgrid(x_b, y_b)

   projection  = TARGET_PROJ_PYPROJ
   lon_b, lat_b = projection(x_b_2d, y_b_2d, inverse=True)

   ds = ds.assign_coords(
      x_b=("x_b", x_b), 
      y_b=("y_b", y_b),
      lon_b=(["y_b", "x_b"], lon_b),
      lat_b=(["y_b", "x_b"], lat_b)
   )
   
   return ds



def interpolation_target_grid(ds):
   ds_target = xr.open_dataset(TARGET_GRID_FILE)
   ds_target = add_lon_lat_bounds(ds_target)

   for i, coord in enumerate(['lat','lon']):
      if len(ds[coord].dims) == 1:
         if len(ds[coord].values) > TARGET_SIZE[i]: # if resolution is finer than target's
               new_coord = np.linspace(ds[coord].values.min(), ds[coord].values.max(), TARGET_SIZE[i])
               ds = ds.interp({coord:new_coord})
   for var in ds.data_vars:
      ds[var].values = np.asfortranarray(ds[var].values)
      ds[var].values = np.ascontiguousarray(ds[var].values)
   
   regridder = xe.Regridder(ds, ds_target, "conservative")
   ds_out = regridder(ds)
   return ds_out


def reformat_as_target(ds):
    ''' Returns Input dataset interpolated at target target grid '''
    ds = standardize_longitudes(ds)
    ds = ds.sel(lon=slice(LONMIN,LONMAX), lat=slice(LATMIN, LATMAX))
    ds = interpolation_target_grid(ds)
    return ds


if __name__=='__main__':
   date_str = '1984-01-01'
   date = pd.Timestamp(date_str).date()
   
   '''
   #ds = xr.open_dataset('/gpfs-calypso/scratch/globc/garcia/rawdata/cmip6/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r10i1p1f2_gr_18500101-20141231.nc')
   ds = xr.open_dataset('/gpfs-calypso/scratch/globc/garcia/rawdata/cmip6/CNRM-CM6-1/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc')
   ds = standardize_longitudes(ds)
   ds = ds.sel(lon=slice(-6,12), lat=slice(40.,52.))
   condition = ds['sftlf'].values < 5
   mask_var_array = ma.masked_array(ds['sftlf'].values, condition)
   plot_image(condition, f'LR Land/Sea mask (< 5 %)', '/scratch/globc/garcia/graph/mask_LR_5.png')


   '''
   ds = xr.open_dataset('/gpfs-calypso/scratch/globc/garcia/rawdata/cmip6/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r10i1p1f2_gr_18500101-20141231.nc')
   ds = standardize_longitudes(ds)
   ds = ds.sel(time=ds.time.dt.date == date)
   ds = ds.isel(time=0)
   ds = ds.sel(lon=slice(-6,12), lat=slice(40.,52.))
   #ds_out = interpolation_target_grid(ds)
   plot_contour(ds['tas'].values, date_str, '/scratch/globc/garcia/graph/test/test.png')
   print('ok')
   '''
   ds_s = xr.open_dataset('/gpfs-calypso/scratch/globc/garcia/rawdata/safran/SAFRAN_2014080107_2015080106_reformat.nc')
   ds_s = ds_s.sel(time=pd.date_range(start=date_str, periods = 23, freq='h').to_numpy())
   print(np.shape(ds_s['tas'].values.mean(axis=0)))
   plot_image(ds_s['tas'].values.mean(axis=0), date_str, '/scratch/globc/garcia/graph/safranHR.png')

   
   ds = xr.open_dataset('/gpfs-calypso/scratch/globc/garcia/utils/ETOPO_2022_v1_30s_N90W180_bed_regrid.nc')
   plot_image(ds['z'].values, 'topography', '/scratch/globc/garcia/graph/topography.png')'''