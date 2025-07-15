from pathlib import Path
import pandas as pd
import cartopy.crs as ccrs
import pyproj



RAW_DIR = Path('/scratch/globc/garcia/rawdata/')
SAFRAN_DIR = RAW_DIR / 'safran'
SAFRAN_RAW_DIR = SAFRAN_DIR / 'raw_safran'
SAFRAN_REFORMAT_DIR = SAFRAN_DIR / 'safran_reformat_day'
GCM_RAW_DIR = RAW_DIR / 'gcm'
RCM_RAW_DIR = RAW_DIR / 'rcm'
ERA5_DIR = RAW_DIR / "era5"
EOBS_RAW_DIR = RAW_DIR / 'eobs'
ALADIN_RAW_DIR = RAW_DIR / 'ALADIN'
TARGET_SAFRAN_FILE = SAFRAN_REFORMAT_DIR / 'tas_day_SAFRAN_1959_reformat.nc'
TARGET_EOBS_FRANCE_FILE = EOBS_RAW_DIR / 'tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc'
TARGET_GCM_FILE = GCM_RAW_DIR / 'CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc'
OROG_EOBS_FRANCE_FILE = EOBS_RAW_DIR / 'elevation_ens_025deg_reg_v29_0e_france.nc'
OROG_SAFRAN_FILE = RAW_DIR / 'topography/topography_safran2.nc'
IMERG_MASK = RAW_DIR / 'landseamask/IMERG_land_sea_mask_regrid.nc' # only continents
LANDSEAMASK_GCM = GCM_RAW_DIR / 'CNRM-CM6-1/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc'
LANDSEAMASK_ERA5 = ERA5_DIR / 'lsm_ERA5.nc'
LANDSEAMASK_EOBS = EOBS_RAW_DIR / 'eobs_landseamask.nc'
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
DATASET_EXP3_30Y_DIR = DATASET_DIR / 'dataset_exp3_30y'
DATASET_EXP3_BASELINE_DIR = DATASET_DIR / 'dataset_exp3_baseline'
DATASET_EXP4_30Y_DIR = DATASET_DIR / 'dataset_exp4_30y'
DATASET_EXP4_BASELINE_DIR = DATASET_DIR / 'dataset_exp4_baseline'
DATASET_EXP5_30Y_DIR = DATASET_DIR / 'dataset_exp5_30y'
DATASET_EXP6_30Y_DIR = DATASET_DIR / 'dataset_exp6_30y'
DATASET_EXP6_BASELINE_DIR = DATASET_DIR / 'dataset_exp6_baseline'
DATASET_TEST_ERA5_DIR = DATASET_DIR / 'dataset_test_era5'
DATASET_TEST_6MB_ISAFRAN = DATASET_DIR / 'dataset_test_6mb_iSAFRAN'
DATASET_BC_DIR = DATASET_DIR / 'dataset_bc'
DATASET_BC_GCM_ERA5 = DATASET_BC_DIR / 'dataset_bc_era5_gcm.npz' # Historical data for bias correction


RUNS_DIR = Path('/scratch/globc/garcia/runs/')
GRAPHS_DIR = Path('/scratch/globc/garcia/graph/')
METRICS_DIR = Path('/scratch/globc/garcia/metrics/')
PREDICTION_DIR = Path('/scratch/globc/garcia/prediction/')

# Plot standard
## FRANCE DOMAIN 

CONFIG = {
    'exp3':
        {'target':'safran',
        'domain': [-6., 12., 40., 52.],  # safran dataset original crop + padding --> 160*160 crop
        'domain_xy' : [60000, 1196000, 1617000, 2681000],
        'data_projection' : ccrs.LambertConformal(central_longitude=2.337229,
                                 central_latitude=46.8,
                                 false_easting=600000,
                                 false_northing=2200000,
                                 standard_parallels=(45.89892, 47.69601)),
        'fig_projection' :ccrs.LambertConformal(central_latitude=46., central_longitude=2.),
        'pyproj_projection' : pyproj.Proj("+proj=lcc +lon_0=2.337229 +lat_0=46.8 +lat_1=45.89892 +lat_2=47.69601 +x_0=600000 +y_0=2200000"),
        'shape' : (134,143),
        'target_file' : OROG_SAFRAN_FILE,
        'orog_file' : OROG_SAFRAN_FILE, # fichier et coordonnées cibles
        'dataset' : DATASET_EXP3_30Y_DIR,
        'target_vars': ['tas'],
        'input_vars': ['elevation', 'tas'],
        'channels': ['topography', 'tas input', 'tas target']
        },

    'exp4':
        {'target':'eobs',
        'domain': 
            {'france': [-6., 10., 38, 54],
             'europe' : [-12.5, 27.5, 31., 71.],
             'tchequie' : [11.5, 19.5, 45.75, 53.75]},
        'data_projection' : ccrs.PlateCarree(),
        'fig_projection' : 
            {'france' : ccrs.LambertConformal(central_latitude=46., central_longitude=2.),
             'europe' : ccrs.LambertConformal(central_latitude=51., central_longitude=7.5),
             'tchequie' : ccrs.LambertConformal(central_latitude=45.75, central_longitude=11.5)},
        'shape': 
                {'france': (64,64),
                'europe' : (160,160),
                'tchequie' : (32,32)},
        'target_file' : EOBS_RAW_DIR / 'tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc',
        'orog_file' : EOBS_RAW_DIR / 'elevation_ens_025deg_reg_v29_0e_france.nc',
        'dataset' : DATASET_EXP4_30Y_DIR,
        'target_vars': ['tas'],
        'input_vars': ['elevation', 'tas'],
        'channels': ['topography', 'tas input', 'tas target']       
        },
    'exp5':
        {'target':'eobs',
        'domain': [-6., 10., 38, 54],
        'data_projection' : ccrs.PlateCarree(),
        'fig_projection' : ccrs.LambertConformal(central_latitude=46., central_longitude=2.),
        'pyproj_projection' : None,
        'shape': (64,64),
        'target_file' : TARGET_EOBS_FRANCE_FILE,
        'orog_file' : OROG_EOBS_FRANCE_FILE,
        'dataset' : DATASET_EXP5_30Y_DIR,
        'target_vars': ['tas'],
        'input_vars': ['elevation', 'tas'],
        'channels': ['topography', 'tas input', 'tas target'],
        'ssp': 'ssp585'        
        },
    'exp6':
        {'target':'eobs',
        'domain': [-6., 10., 38, 54],
        'data_projection' : ccrs.PlateCarree(),
        'fig_projection' : ccrs.LambertConformal(central_latitude=46., central_longitude=2.),
        'pyproj_projection' : None,
        'shape': (64,64),
        'target_file' : TARGET_EOBS_FRANCE_FILE,
        'orog_file' : OROG_EOBS_FRANCE_FILE,
        'dataset' : DATASET_EXP6_30Y_DIR,
        'target_vars': ['pr'],
        'input_vars': ['elevation', 'pr'],
        'channels': ['topography', 'pr input', 'pr target'],
        'ssp' : 'ssp585'   
        }
    }

COLORS = {'SAFRAN 8km': 'purple',
          'E-OBS 25km' : 'blue',
          'ERA5 8km' : 'cyan',
          'ERA5 0.25°' : 'green',
          'GCM 1°' : 'orange',
          'RCM 12km' : 'orange',
          'UNet' : 'orangered',
          'SwinUNETR' : 'hotpink'}

ALADIN_PROJ_PYPROJ = pyproj.Proj("+proj=lcc +lat_1=49.500000 +lat_0=49.500000 +lon_0=10.500000 +k_0=1.0 +x_0=2925000.000000 +y_0=2925000.000000 +R=6371229.000000", preserve_units=True)
SAFRAN_PROJ_PYPROJ = pyproj.Proj("+proj=lcc +lon_0=2.337229 +lat_0=46.8 +lat_1=45.89892 +lat_2=47.69601 +x_0=600000 +y_0=2200000")
SAFRAN_PROJ = ccrs.LambertConformal(central_longitude=2.337229,
                                 central_latitude=46.8,
                                 false_easting=600000,
                                 false_northing=2200000,
                                 standard_parallels=(45.89892, 47.69601))


# Experience settings
#DATES = pd.date_range(start='1985-01-01', end='2014-12-31', freq='D') # exp3 exp4
DATES = pd.date_range(start='1980-01-01', end='2014-12-31', freq='D') # all data

GCM = ['CNRM-CM6-1']

# Test settings
#DATES_TEST = pd.date_range(start='2012-10-18', end='2014-12-31', freq='D') #exp1 exp2
DATES_TEST = pd.date_range(start='2010-01-01', end='2014-12-31', freq='D') # exp3


DATES_BC_TRAIN_HIST = pd.date_range(start='1980-01-01', end='1999-12-31', freq='D')
DATES_BC_TEST_HIST = pd.date_range(start='2000-01-01', end='2014-12-31', freq='D')
DATES_BC_TEST_FUTURE = pd.date_range(start='2015-01-01', end='2100-12-31', freq='D')

