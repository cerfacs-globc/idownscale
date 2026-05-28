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
    ALADIN_PROJ_PYPROJ,
    CONFIG,
    DATASET_BC_DIR,
    DATES_BC_TEST_FUTURE,
    DATES_BC_TEST_HIST,
    DATES_BC_TRAIN_HIST,
    get_bias_corrected_netcdf_path,
    get_simu_source,
)
from iriscc.datautils import Data, reformat_as_target


def dataset_variant_dir(exp: str, variant: str) -> Path:
    return DATASET_BC_DIR / f"dataset_{exp}_test_{variant}"


def source_period_date(period: str):
    if period == "train_hist":
        return DATES_BC_TRAIN_HIST[0]
    if period == "test_hist":
        return DATES_BC_TEST_HIST[0]
    if period == "test_future":
        return DATES_BC_TEST_FUTURE[0]
    raise ValueError(f"Unsupported period: {period}")


def simu_netcdf_path(get_data: Data, exp: str, simu: str, var: str, ssp: str, corrected: bool, period: str) -> Path:
    if corrected:
        return get_bias_corrected_netcdf_path(exp, simu, var, period, ssp=ssp)
    source_name = get_simu_source(exp, simu)
    return Path(get_data._resolve_source_file(source_name, var, date=source_period_date(period), ssp=ssp))


def select_date(ds: xr.Dataset, date) -> xr.Dataset:
    ds_i = ds.sel(time=ds.time.dt.date == date.date())
    return ds_i.isel(time=0, drop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build daily inference sample datasets from GCM fields")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp5)', default='exp5')
    parser.add_argument('--var', type=str, help='Variable to use', default='tas')
    parser.add_argument('--simu', type=str, help='Simulation family', default='gcm')
    parser.add_argument('--ssp', type=str, default=None, help='Scenario override. Defaults to experiment config.')
    parser.add_argument('--corrected', action='store_true', help='Use bias-corrected simulation files and write dataset_<exp>_test_<simu>_bc')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional explicit output dataset directory')
    parser.add_argument('--test', action='store_true', help='Only build the first day from each period')
    args = parser.parse_args()

    exp = args.exp
    var = args.var
    ssp = args.ssp or CONFIG[exp]['ssp']
    domain = CONFIG[exp]['domain']
    orog_file = CONFIG[exp]['orog_file']
    target_file = CONFIG[exp]['target_file']
    variant = f"{args.simu}_bc" if args.corrected else args.simu
    output_dir = Path(args.output_dir) if args.output_dir else dataset_variant_dir(exp, variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    get_data = Data(domain=domain)
    source_name = get_simu_source(exp, args.simu)
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
        ds_cache[period] = xr.open_dataset(simu_netcdf_path(get_data, exp, args.simu, var, ssp, args.corrected, period))
        if not args.corrected:
            ds_cache[period] = get_data._standardize_source_geometry(
                ds_cache[period],
                get_data.get_source_spec(source_name).get("geometry", "none"),
            )
        for date in dates:
            print(date)
            ds_i = select_date(ds_cache[period], date)
            ds_i = reformat_as_target(
                ds_i,
                target_file=target_file,
                domain=domain,
                method="conservative_normed",
                mask=True,
                input_projection=ALADIN_PROJ_PYPROJ if (not args.corrected and source_name == "rcm_aladin") else None,
            )

            x = np.stack([orog, ds_i[var].values], axis=0)
            sample = {'x': x.astype(np.float32)}

            if date <= DATES_BC_TEST_HIST[-1]:
                ds_target = get_data.get_target_dataset(
                    target=CONFIG[exp]['target'],
                    var=var,
                    date=date,
                    source_name=CONFIG[exp].get('target_source'),
                )
                y = ds_target[var].values
                sample['y'] = np.expand_dims(y, axis=0).astype(np.float32)

            np.savez(output_dir / f"sample_{date.strftime('%Y%m%d')}.npz", **sample)
