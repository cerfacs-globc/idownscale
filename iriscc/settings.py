"""
Experimental settings for all configurations.

date : 16/07/2025
author : Zoé GARCIA

date : 18/02/2026
modifications: Christian Pagé

"""

from pathlib import Path
import pandas as pd
import sys
import os
import cartopy.crs as ccrs
import pyproj

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv('IDOWNSCALE_DATA_DIR', Path.cwd()))
RAW_DIR = Path(os.getenv('IDOWNSCALE_RAW_DIR', REPO_DIR / 'rawdata'))
LOCAL_RAW_DIR = RAW_DIR
SAFRAN_DIR = RAW_DIR / 'safran'
SAFRAN_RAW_DIR = SAFRAN_DIR / 'raw_safran'
SAFRAN_REFORMAT_DIR = SAFRAN_DIR / 'safran_reformat_day'
GCM_RAW_DIR = RAW_DIR / 'gcm'
RCM_RAW_DIR = RAW_DIR / 'rcm'
ERA5_DIR = RAW_DIR / 'era5'
EOBS_RAW_DIR = RAW_DIR / 'eobs'
LOCAL_EOBS_RAW_DIR = LOCAL_RAW_DIR / 'eobs'
ALADIN_RAW_DIR = RAW_DIR / 'ALADIN'
CERRA_RAW_DIR = RAW_DIR / 'cerra'
TARGET_SAFRAN_FILE = SAFRAN_REFORMAT_DIR / 'tas_day_SAFRAN_1959_reformat.nc'
TARGET_EOBS_FRANCE_FILE = EOBS_RAW_DIR / 'tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc'
TARGET_GCM_FILE = GCM_RAW_DIR / 'CNRM-CM6-1/historical/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc'
OROG_EOBS_FRANCE_FILE = LOCAL_EOBS_RAW_DIR / 'elevation_ens_025deg_reg_v29_0e_france.nc'
OROG_SAFRAN_FILE = RAW_DIR / 'topography/topography_safran.nc'
IMERG_MASK = RAW_DIR / 'landseamask/IMERG_land_sea_mask_regrid.nc' # only continents
LANDSEAMASK_GCM = GCM_RAW_DIR / 'sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc'
LANDSEAMASK_ERA5 = ERA5_DIR / 'lsm_ERA5.nc'
LANDSEAMASK_EOBS = EOBS_RAW_DIR / 'eobs_landseamask.nc'
COUNTRIES_MASK = RAW_DIR / 'landseamask/CNTR_RG_10M_2024_4326.nc'
UTILS_DIR = Path(os.getenv('IDOWNSCALE_UTILS_DIR', '/scratch/globc/garcia/utils/'))
DEFAULT_GRID_FILE = UTILS_DIR / 'tasmax_1d_21000101_21001231.nc'

DATASET_DIR = DATA_DIR / 'datasets'
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
DATASET_EXP7_30Y_DIR = DATASET_DIR / 'dataset_exp7_30y'
DATASET_EXP8_30Y_DIR = DATASET_DIR / 'dataset_exp8_30y'
DATASET_TEST_ERA5_DIR = DATASET_DIR / 'dataset_test_era5'
DATASET_BC_DIR = DATASET_DIR / 'dataset_bc'

RUNS_DIR = DATA_DIR / 'runs'
GRAPHS_DIR = DATA_DIR / 'graph'
METRICS_DIR = DATA_DIR / 'metrics'
PREDICTION_DIR = DATA_DIR / 'prediction'
OUTPUT_DIR = DATA_DIR / 'output'

DATASET_METADATA = {
    'era5': {
        'var_map': {'pr': 'tp'},
        'file_pattern': '{var}*_{year}_*',
        'dir_pattern': '{var}_1d'
    },
    'cerra': {
        'var_map': {'tas': 'tas', 'pr': 'tp'}, 
        'file_pattern': '{var}*'
    },
    'safran': {
        'var_map': {'tas': 'tas', 'pr': 'pr'},
        'file_pattern': '{var}*{year}_reformat.nc'
    },
    'eobs': {
        'var_map': {'tas': 'tas', 'pr': 'pr'},
        'file_pattern': '{var}*'
    },
    'gcm': {
        'var_map': {'tas': 'tas', 'pr': 'pr'},
        'file_pattern': '*/{var}*{period}*r1i1p1f2*'
    },
    'rcm': {
        'var_map': {'tas': 'tas', 'pr': 'pr'},
        'file_pattern': 'ALADIN/{var}*{period}*r1i1p1f2*'
    }
}

CONFIG = {
    'exp3':
        {'target': 'safran',
            'domain': [-6., 12., 40., 52.],
            'domain_xy' : [60000, 1196000, 1617000, 2681000],
            'data_projection' : ccrs.LambertConformal(central_longitude=2.337229,
                                    central_latitude=46.8,
                                    false_easting=600000,
                                    false_northing=2200000,
                                    standard_parallels=(45.89892, 47.69601)),
            'pyproj_projection': pyproj.Proj("+proj=lcc +lon_0=2.337229 +lat_0=46.8 +lat_1=45.89892 +lat_2=47.69601 +x_0=600000 +y_0=2200000"),
            'shape': (134, 143),
            'target_file': OROG_SAFRAN_FILE,
            'orog_file': OROG_SAFRAN_FILE, 
            'dataset': DATASET_EXP3_30Y_DIR,
            'target_vars': ['tas'],
            'input_vars': ['elevation', 'tas'],
            'channels': ['elevation', 'tas input', 'tas target'],
            'remove_countries': True,
            'domain_name': 'france',
            'freq': '1D'
        },

    'exp4': # obsolete, use exp5 
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
            'shape': {
                'france': (64, 64),
                'europe': (160, 160),
                'tchequie': (32, 32)
            },
            'target_file' : EOBS_RAW_DIR / 'tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc',
            'orog_file' : EOBS_RAW_DIR / 'elevation_ens_025deg_reg_v29_0e_france.nc',
            'dataset' : DATASET_EXP4_30Y_DIR,
            'target_vars': ['tas'],
            'input_vars': ['elevation', 'tas'],
            'channels': ['elevation', 'tas input', 'tas target'],
            'domain_name': 'france'
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
            'channels': ['elevation', 'tas input', 'tas target'],
            'ssp': 'ssp585',
            'mask': 'target',
            'freq': '1D',
            'fill_value': 0.0,
            'output_norm': True,
            'model': 'unet',
            'debiaser': 'cdft',
            'ai_step': True,
            'domain_name': 'france'
        },
    'exp6':
        {'target':'eobs',
            'domain': [-6., 10., 38, 54],
            'data_projection' : ccrs.PlateCarree(),
            'fig_projection' : ccrs.LambertConformal(central_latitude=46., central_longitude=2.),
            'pyproj_projection' : None, # for curvilign grids conservative interpolation
            'shape': (64,64),
            'target_file' : TARGET_EOBS_FRANCE_FILE, # target grid coordinates
            'orog_file' : OROG_EOBS_FRANCE_FILE,
            'dataset' : DATASET_EXP6_30Y_DIR,
            'target_vars': ['pr'],
            'input_vars': ['elevation', 'pr'],
            'channels': ['elevation', 'pr input', 'pr target'], # to not get lost for normalization
            'ssp': 'ssp585',
            'debiaser': 'cdft',
            'model': 'unet',
            'ai_step': True,
            'input_source': 'era5',
            'remove_countries': False,
            'domain_name': 'france'
        },
    'exp7':
        {'target':'eobs',
            'domain': [-6., 10., 38, 54],
            'data_projection' : ccrs.PlateCarree(),
            'fig_projection' : ccrs.LambertConformal(central_latitude=46., central_longitude=2.),
            'pyproj_projection' : None, # for curvilign grids conservative interpolation
            'shape': (64,64),
            'target_file' : TARGET_EOBS_FRANCE_FILE, # target grid coordinates
            'orog_file' : OROG_EOBS_FRANCE_FILE,
            'dataset' : DATASET_EXP7_30Y_DIR,
            'target_vars': ['pr'],
            'input_vars': ['elevation', 'huss', 'psl', 'tas'],
            'channels': ['elevation', 'huss input', 'psl input', 'tas input', 'pr target'], # to not get lost for normalization
            'ssp': 'ssp585',
            'input_source': 'era5',
            'remove_countries': False,
            'domain_name': 'france'
        },
    'exp8':
        {'target':'eobs',
            'domain': [-6., 10., 38, 54],
            'data_projection' : ccrs.PlateCarree(),
            'fig_projection' : ccrs.LambertConformal(central_latitude=46., central_longitude=2.),
            'pyproj_projection' : None, # for curvilign grids conservative interpolation
            'shape': (64,64),
            'target_file' : TARGET_EOBS_FRANCE_FILE, # target grid coordinates
            'orog_file' : OROG_EOBS_FRANCE_FILE,
            'dataset' : DATASET_EXP8_30Y_DIR,
            'target_vars': ['pr'],
            'input_vars': ['elevation',
                        'zg500',
                        'zg700',
                        'zg850',
                        'ta500',
                        'ta700',
                        'ta850',
                        'ua500',
                        'ua700',
                        'ua850',
                        'vas',
                        'uas',
                        'psl'],
            'channels': ['elevation',
                        'zg500 input',
                        'zg700 input',
                        'zg850 input',
                        'ta500 input',
                        'ta700 input',
                        'ta850 input',
                        'ua500 input',
                        'ua700 input',
                        'ua850 input',
                        'vas input',
                        'uas input',
                        'psl input',
                        'pr target'], # to not get lost for normalization
            'ssp': 'ssp585',
            'input_source': 'era5',
            'remove_countries': False,
            'domain_name': 'france'
        },

    'exp_cerra_test': {
        'target': 'cerra',
        'domain': [-6., 10., 38, 54],
        'data_projection' : ccrs.PlateCarree(),
        'fig_projection' : ccrs.LambertConformal(central_latitude=46., central_longitude=2.),
        'pyproj_projection' : None,
        'shape': (64,64),
        'target_file' : TARGET_EOBS_FRANCE_FILE,
        'orog_file' : OROG_EOBS_FRANCE_FILE,
        'dataset' : DATASET_DIR / 'dataset_cerra_test',
        'target_vars': ['tas'],
        'input_vars': ['elevation', 'tas'],
        'channels': ['elevation', 'tas input', 'tas target'],
        'ssp': 'ssp585',
        'debiaser': 'cdft',
        'model': 'unet',
        'ai_step': True,
        'input_source': 'era5',
        'remove_countries': False,
        'domain_name': 'france'
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

ALADIN_PROJ_PYPROJ = pyproj.Proj(
    "+proj=lcc +lat_1=49.500000 +lat_0=49.500000 +lon_0=10.500000 +k_0=1.0 +x_0=2925000.000000 +y_0=2925000.000000 +R=6371229.000000", preserve_units=True)
SAFRAN_PROJ_PYPROJ = pyproj.Proj(
    "+proj=lcc +lon_0=2.337229 +lat_0=46.8 +lat_1=45.89892 +lat_2=47.69601 +x_0=600000 +y_0=2200000")


# Phase 1 settings
DATES = pd.date_range(start='19850101', end='2004-12-31', freq='D')
DATES_TRAIN = ['1985', '2001', '2003'] # train, valid, test start (ex8 mini dataset fior test)
# DATES_TRAIN = ['1985', '2004', '2010'] # train, valid, test start
DATES_TEST = pd.date_range(start='2010-01-01', end='2014-12-31', freq='D') 

# Phase 2 settings
# DATES = pd.date_range(start='1980-01-01', end='2014-12-31', freq='D') # all data for phase 2
# DATES_TRAIN = ['1980', '2010', '2014'] # train, valid, test start

DATES_BC_TRAIN_HIST = pd.date_range(start='1980-01-01', end='1999-12-31', freq='D')
DATES_BC_TEST_HIST = pd.date_range(start='2000-01-01', end='2014-12-31', freq='D')
DATES_BC_TEST_FUTURE = pd.date_range(start='2015-01-01', end='2100-12-31', freq='D')

