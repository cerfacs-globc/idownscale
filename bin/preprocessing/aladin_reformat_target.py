import sys
sys.path.append('.')

import glob
import xarray as xr
import numpy as np
import argparse
import pandas as pd


from iriscc.settings import (TARGET_EOBS_FRANCE_FILE, 
                             CONFIG, 
                             RCM_RAW_DIR, 
                             ALADIN_PROJ_PYPROJ,
                             DATES_BC_TEST_HIST,
                             DATES_BC_TEST_FUTURE
                             )

from iriscc.datautils import reformat_as_target, standardize_longitudes, Data
from bin.training.predict_loop import get_target_format

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Reformat RCM to target for comparison ")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')
    parser.add_argument('--var', type=str, help='Variable to reformat', default='tas') 
    args = parser.parse_args()
    exp = args.exp
    var = args.var

    get_data = Data(domain=CONFIG[exp]['domain'])
    '''
    ds_test_hist, _ = get_target_format(exp, DATES_BC_TEST_HIST)
    for i, date in enumerate(DATES_BC_TEST_HIST):
        print(date)
        ds = get_data.get_rcm_dataset(var, date.date())
        ds = reformat_as_target(ds, 
                        target_file=CONFIG[exp]['target_file'], 
                        method='bilinear', 
                        domain=CONFIG[exp]['domain'], 
                        mask=True, 
                        crop_input=False,
                        crop_target=False, 
                        input_projection=ALADIN_PROJ_PYPROJ,
                        target_projection=CONFIG[exp]['pyproj_projection'])
        
        ds_test_hist[var][i] = ds[var].values
    ds_test_hist.to_netcdf(RCM_RAW_DIR/f'ALADIN_reformat/tas_EUR-12_CNRM-ESM2-1_historical_r1i1p1f2_CNRM-MF_CNRM-ALADIN64E1_v1-r1_day_20000101-20141231_reformat_E-OBS_france.nc')
    '''
    dates = pd.date_range(start='1980-01-01', end='1999-12-31', freq='D')
    ds_test_future, _ = get_target_format(exp, dates)
    for i, date in enumerate(dates):
        print(date)
        ds = get_data.get_rcm_dataset(var, date)
        ds = reformat_as_target(ds, 
                        target_file=CONFIG[exp]['target_file'], 
                        method='bilinear', 
                        domain=CONFIG[exp]['domain'], 
                        mask=True, 
                        crop_input=False,
                        crop_target=False, 
                        input_projection=None,
                        target_projection= CONFIG[exp]['pyproj_projection'])
        
        ds_test_future[var][i] = ds[var].values
    ds_test_future.to_netcdf(RCM_RAW_DIR/f'ALADIN_reformat/tas_EUR-12_CNRM-ESM2-1_historical_r1i1p1f2_CNRM-MF_CNRM-ALADIN64E1_v1-r1_day_19800101-19991231_reformat_E-OBS_france.nc')


