import sys
sys.path.append('.')
import xarray as xr


from iriscc.datautils import reformat_as_target
from iriscc.settings import TARGET_GRID_FILE


''' Create a new netcdf file regridded at target grid'''

if __name__== '__main__':

    file = str(sys.argv[1])
 
    ds = xr.open_dataset(file, engine='netcdf4')
    for var in ds.data_vars:
        ds[var] = ds[var].transpose()
    
    ds = reformat_as_target(ds, target_file=TARGET_GRID_FILE)
    ds.to_netcdf(f'{file[:-3]}_regrid.nc')