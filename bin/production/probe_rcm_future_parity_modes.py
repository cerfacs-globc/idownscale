import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bin.preprocessing.build_dataset import Data
from iriscc.datautils import crop_domain_from_ds, interpolation_target_grid
from iriscc.settings import ALADIN_PROJ_PYPROJ, CONFIG, get_simu_source


def parse_args():
    parser = argparse.ArgumentParser(description="Compare RCM remap modes against archive for selected future dates.")
    parser.add_argument("--exp", default="exp5")
    parser.add_argument("--simu", default="rcm")
    parser.add_argument("--var", default="tas")
    parser.add_argument("--ssp", default="ssp585")
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--dates", nargs="+", required=True)
    return parser.parse_args()


def select_daily_window(ds, date):
    stop = date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    ds = ds.sel(time=slice(date, stop))
    ds = ds.resample(time="1D").mean()
    ds = ds.sel(time=[date])
    return ds


def load_batch_dataset(get_bc_data, source_name, var, date, ssp, *, domain_override=None):
    spec = get_bc_data.get_source_spec(source_name)
    ds = get_bc_data._open_source_dataset(source_name, var, date=date, ssp=ssp)
    ds = select_daily_window(ds, date)
    if spec.get("geometry") != "rcm":
        ds = crop_domain_from_ds(ds, domain_override if domain_override is not None else get_bc_data.domain)
    ds[var].values = get_bc_data.clean_data(ds[var].values, var, data_type=spec.get("data_type"))
    return ds


def strip_native_bounds(ds):
    ds = ds.copy()
    for name in ("bounds_lon", "bounds_lat", "lon_bounds", "lat_bounds"):
        if name in ds.variables:
            ds = ds.drop_vars(name)
    if "lon" in ds.coords:
        ds["lon"].attrs.pop("bounds", None)
    if "lat" in ds.coords:
        ds["lat"].attrs.pop("bounds", None)
    return ds


def stats(arr, ref):
    diff = arr - ref
    return {
        "mad": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs": float(np.max(np.abs(diff))),
        "signed_mean": float(np.mean(diff)),
    }


def main():
    args = parse_args()
    exp = args.exp
    domain = CONFIG[exp]["domain"]
    bc_domain = CONFIG[exp].get("bc_domain", domain)
    gcm_source = CONFIG[exp].get("gcm_source", "gcm_cnrm_cm6_1")
    simu_source = get_simu_source(exp, args.simu)
    get_bc_data = Data(domain=bc_domain)

    archive_path = Path(args.archive_root) / "datasets" / "dataset_bc" / f"bc_test_future_{args.simu}.npz"
    archive = np.load(archive_path, allow_pickle=True)
    archive_dates = pd.to_datetime(archive["dates"])

    for raw_date in args.dates:
        date = pd.Timestamp(raw_date)
        print(f"DATE {date.date()}", flush=True)

        ds_simu = load_batch_dataset(get_bc_data, simu_source, args.var, date, args.ssp)
        ds_gcm = load_batch_dataset(get_bc_data, gcm_source, args.var, date, args.ssp)
        target_grid = ds_gcm.isel(time=0, drop=True)
        target_grid_domain = crop_domain_from_ds(target_grid, domain)

        native = interpolation_target_grid(
            ds_simu.copy(),
            ds_target=target_grid,
            method="conservative_normed",
            input_projection=ALADIN_PROJ_PYPROJ,
        )
        synthetic = interpolation_target_grid(
            strip_native_bounds(ds_simu),
            ds_target=target_grid,
            method="conservative_normed",
            input_projection=ALADIN_PROJ_PYPROJ,
        )
        bilinear = interpolation_target_grid(
            ds_simu.copy(),
            ds_target=target_grid,
            method="bilinear",
            input_projection=ALADIN_PROJ_PYPROJ,
        )
        legacy_daily_simu = get_bc_data.get_model_dataset(simu_source, args.var, date, ssp=args.ssp)
        legacy_daily_gcm = get_bc_data.get_model_dataset(gcm_source, args.var, date, ssp=args.ssp)
        legacy_daily = interpolation_target_grid(
            legacy_daily_simu.copy(),
            ds_target=legacy_daily_gcm,
            method="conservative_normed",
            input_projection=ALADIN_PROJ_PYPROJ,
        )

        index = int(np.where(archive_dates == np.datetime64(date))[0][0])
        ref = archive[args.simu][index]

        for label, arr in (
            ("native_conservative", native[args.var].values[0]),
            ("legacy_daily_native", legacy_daily[args.var].values[0]),
            ("synthetic_conservative", synthetic[args.var].values[0]),
            ("bilinear", bilinear[args.var].values[0]),
        ):
            result = stats(arr, ref)
            print(
                f"  {label}: mad={result['mad']:.9f} rmse={result['rmse']:.9f} "
                f"max_abs={result['max_abs']:.9f} signed_mean={result['signed_mean']:.9f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
