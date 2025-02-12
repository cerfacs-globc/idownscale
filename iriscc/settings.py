from pathlib import Path
import pandas as pd
import cartopy.crs as ccrs
import pyproj

SAFRAN_RAW_DIR = Path('/archive2/globc/dcom/globc_obs/SAFRAN/')

RAW_DIR = Path('/scratch/globc/garcia/rawdata/')
SAFRAN_DIR = RAW_DIR / 'safran'
CMIP6_RAW_DIR = RAW_DIR / 'cmip6'
ERA5_DIR = RAW_DIR / "era5"
TARGET_GRID_FILE = SAFRAN_DIR / 'SAFRAN_1958080107_1959080106_reformat.nc'
OROG_FILE = RAW_DIR / 'topography/ETOPO_2022_v1_30s_N90W180_bed_regrid.nc'
IMERG_MASK= RAW_DIR / 'landseamask/IMERG_land_sea_mask_regrid.nc'
COUNTRIES_MASK = RAW_DIR /'landseamask/CNTR_RG_10M_2024_4326.nc'


DATASET_DIR = Path('/scratch/globc/garcia/datasets/')
DATASET_EXP1_DIR = DATASET_DIR / 'dataset_exp1'
DATASET_EXP1_CONTINENTS_DIR = DATASET_DIR / 'dataset_exp1_continents'
DATASET_EXP1_30Y_DIR = DATASET_DIR / 'dataset_exp1_30y'
DATASET_EXP1_6MB_DIR = DATASET_DIR / 'dataset_exp1_6mb'
DATASET_EXP1_6MB_30Y_DIR = DATASET_DIR / 'dataset_exp1_6mb_30y'
DATASET_EXP2_DIR = DATASET_DIR / 'dataset_exp2'
DATASET_EXP2_6MB_DIR = DATASET_DIR / 'dataset_exp2_6mb'
DATASET_EXP2_BI_DIR = DATASET_DIR / 'dataset_exp2_bi'
DATASET_TEST_ERA5_DIR = DATASET_DIR / 'dataset_test_era5'
DATASET_TEST_6MB_ISAFRAN = DATASET_DIR / 'dataset_test_6mb_iSAFRAN'
DATASET_BC_CMIP6_ERA5 = DATASET_DIR / 'dataset_bc_era5_cmip6.npz' # Historical data for bias correction
DATASET_TEST_BC_CMIP6_DIR = DATASET_DIR / 'dataset_test_bc_cmip6' # Bias corrected dataset


RUNS_DIR = Path('/scratch/globc/garcia/runs/')
GRAPHS_DIR = Path('/scratch/globc/garcia/graph/')
METRICS_DIR = Path('/scratch/globc/garcia/metrics/')

# Plot standard
## FRANCE DOMAIN 
LONMIN = -6.
LONMAX = 12.
LATMIN = 39.
LATMAX = 52.

COORDS_GEO = [-6., 12., 40., 52.]
COORDS_CART = [60000, 1196000, 1617000, 2681000]
TARGET_PROJ_PYPROJ = pyproj.Proj("+proj=lcc +lon_0=2.337229 +lat_0=46.8 +lat_1=45.89892 +lat_2=47.69601 +x_0=600000 +y_0=2200000")
TARGET_PROJ = ccrs.LambertConformal(central_longitude=2.337229,
                                 central_latitude=46.8,
                                 false_easting=600000,
                                 false_northing=2200000,
                                 standard_parallels=(45.89892, 47.69601))
INPUT_PROJ = ccrs.PlateCarree()

DOMAIN = COORDS_CART
PROJ = TARGET_PROJ

# Experience settings
## First experience
#STATISTICS_FILE = DATASET_EXP1_30Y_DIR / 'statistics.json'
#DATES = pd.date_range(start='2012-05-13', end='2014-12-31', freq='D')
DATES = pd.date_range(start='2004-01-01', end='2014-12-31', freq='D')
GCM = ['CNRM-CM6-1']

### Preprocessing
INPUTS = ['tas']
TARGET = 'tas'
CHANELS = ['topography',
           'CNRM-CM6-1 r10i1p1f2']
TARGET_SIZE = [134, 143]
TARGET_MODEL_SIZE =[160, 160]

# Test settings
DATES_TEST = pd.date_range(start='2012-10-18', end='2014-12-31', freq='D')

DATES_BC_FUTURE = pd.date_range(start='2000-01-01', end='2014-12-31', freq='D')
DATES_BC_HIST = pd.date_range(start='1984-01-01', end='1999-12-31', freq='D')


