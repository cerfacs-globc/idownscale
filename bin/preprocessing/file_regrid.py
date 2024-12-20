import sys
sys.path.append('.')
import xesmf as xe


import xarray as xr
import numpy as np


from iriscc.datautils import reformat_as_target


''' Create a new netcdf file regridded at target grid'''

if __name__== '__main__':

    file = str(sys.argv[1])
 
    ds = xr.open_dataset(file, engine='netcdf4')
    for var in ds.data_vars:
        ds[var] = ds[var].transpose()
    
    ds = reformat_as_target(ds)
    ds.to_netcdf(f'{file[:-3]}_regrid.nc')