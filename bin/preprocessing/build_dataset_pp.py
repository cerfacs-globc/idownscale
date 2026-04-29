"""
Build inference-ready daily sample datasets from coarse GCM fields.

This script can package either raw GCM or bias-corrected GCM fields into the
`dataset_bc/dataset_<exp>_test_<variant>` layout used by prediction and metrics
scripts. Historical dates include targets; future dates include inputs only.
"""

import sys
sys.path.append('.')

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from iriscc.settings import (
    CONFIG,
    DATASET_BC_DIR,
    DATES_BC_TEST_FUTURE,
    DATES_BC_TEST_HIST,
    DATES_BC_TRAIN_HIST,
    GCM_BC_DIR,
    GCM_RAW_DIR,
)
from iriscc.datautils import Data, reformat_as_target


def dataset_variant_dir(exp: str, variant: str) -> Path:
    return DATASET_BC_DIR / f"dataset_{exp}_test_{variant}"


def gcm_netcdf_path(var: str, ssp: str, corrected: bool, period: str) -> Path:
    if corrected:
        base = GCM_BC_DIR
    else:
        base = GCM_RAW_DIR / "CNRM-CM6-1"

    if period == "train_hist":
        if corrected:
            return base / f"{var}_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc"
        return base / f"{var}_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc"
    if period == "test_hist":
        if corrected:
            return base / f"{var}_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc"
        return base / f"{var}_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc"
    if period == "test_future":
        if corrected:
            return base / f"{var}_day_CNRM-CM6-1_{ssp}_r1i1p1f2_gr_20150101-21001231_bc.nc"
        return base / f"{var}_day_CNRM-CM6-1_{ssp}_r1i1p1f2_gr_20150101-21001231.nc"
    raise ValueError(f"Unsupported period: {period}")


def select_date(ds: xr.Dataset, date) -> xr.Dataset:
    ds_i = ds.sel(time=ds.time.dt.date == date.date())
    return ds_i.isel(time=0, drop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build daily inference sample datasets from GCM fields")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp5)', default='exp5')
    parser.add_argument('--var', type=str, help='Variable to use', default='tas')
    parser.add_argument('--simu', type=str, help='Simulation family', default='gcm')
    parser.add_argument('--corrected', action='store_true', help='Use bias-corrected GCM files and write dataset_<exp>_test_gcm_bc')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional explicit output dataset directory')
    parser.add_argument('--test', action='store_true', help='Only build the first day from each period')
    args = parser.parse_args()

    if args.simu != 'gcm':
        raise ValueError("build_dataset_pp currently supports only --simu gcm")

    exp = args.exp
    var = args.var
    ssp = CONFIG[exp]['ssp']
    domain = CONFIG[exp]['domain']
    orog_file = CONFIG[exp]['orog_file']
    target_file = CONFIG[exp]['target_file']
    variant = 'gcm_bc' if args.corrected else 'gcm'
    output_dir = Path(args.output_dir) if args.output_dir else dataset_variant_dir(exp, variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    get_data = Data(domain=domain)
    ds_orog = xr.open_dataset(orog_file)
    orog = ds_orog['elevation'].values if 'elevation' in ds_orog else ds_orog['z'].values

    if args.corrected:
        periods = [
            ("train_hist", DATES_BC_TRAIN_HIST),
            ("test_hist", DATES_BC_TEST_HIST),
            ("test_future", DATES_BC_TEST_FUTURE),
        ]
    else:
        periods = [
            ("test_hist", DATES_BC_TEST_HIST),
            ("test_future", DATES_BC_TEST_FUTURE),
        ]

    ds_cache = {}
    for period, dates in periods:
        if args.test:
            dates = dates[:1]
        if len(dates) == 0:
            continue
        ds_cache[period] = xr.open_dataset(gcm_netcdf_path(var, ssp, args.corrected, period))
        for date in dates:
            print(date)
            ds_i = select_date(ds_cache[period], date)
            ds_i = reformat_as_target(
                ds_i,
                target_file=target_file,
                domain=domain,
                method="conservative_normed",
                mask=True,
            )

            x = np.stack([orog, ds_i[var].values], axis=0)
            sample = {'x': x.astype(np.float32)}

            if date <= DATES_BC_TEST_HIST[-1]:
                ds_target = get_data.get_target_dataset(target=CONFIG[exp]['target'], var=var, date=date)
                y = ds_target[var].values
                sample['y'] = np.expand_dims(y, axis=0).astype(np.float32)

            np.savez(output_dir / f"sample_{date.strftime('%Y%m%d')}.npz", **sample)
