"""
Reformat RCM data to match target format for comparison

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import argparse
import pandas as pd


from iriscc.settings import (CONFIG, 
                             RCM_RAW_DIR,
                             DATES_BC_TEST_HIST)

from iriscc.datautils import reformat_as_target, Data
from bin.training.predict_loop import get_target_format

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Reformat RCM to target for comparison ")
    parser.add_argument('--startdate', type=str, help='Start date (e.g., 20000101)', default='20000101')
    parser.add_argument('--enddate', type=str, help='End date (e.g., 20141231)', default='20141231')
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')
    parser.add_argument('--var', type=str, help='Variable to reformat', default='tas') 
    args = parser.parse_args()
    exp = args.exp
    var = args.var
    startdate = args.startdate
    enddate = args.enddate


    get_data = Data(domain=CONFIG[exp]['domain'])
    target = CONFIG[exp]['target']  
    
    dates = pd.date_range(start=startdate, end=enddate, freq='D')
    if dates[-1] < pd.Timestamp('2014-12-31 00:00:00'):
        period = 'historical'
    else:
        period = CONFIG[exp]['ssp']
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
    ds_test_future.to_netcdf(RCM_RAW_DIR/f'ALADIN_reformat/tas_EUR-12_CNRM-ESM2-1_{period}_r1i1p1f2_CNRM-MF_CNRM-ALADIN64E1_v1-r1_day_{startdate}-{enddate}_reformat_{target}.nc')


