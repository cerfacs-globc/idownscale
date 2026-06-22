"""
Build daily sample datasets from coarse simulation fields.

This script can package raw or bias-corrected simulation fields into the
`dataset_bc/dataset_<exp>_...` layouts used by prediction and metrics. Generic
future samples include inputs only; perfect-model runs can explicitly request a
native model source as pseudo-truth for both historical and future targets.
"""

import sys
sys.path.append(".")

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

from iriscc.provenance import build_prov_bundle, inventory_paths, print_resolved_context, utc_now_iso, write_provjson
from iriscc.settings import (
    ALADIN_PROJ_PYPROJ,
    CONFIG,
    DATASET_BC_DIR,
    build_time_range,
    format_sample_time_token,
    get_bc_test_future_dates,
    get_bc_test_hist_dates,
    get_bc_train_hist_dates,
    get_bias_corrected_netcdf_path,
    get_experiment_prediction_frequency,
    get_experiment_training_frequency,
    get_frequency_pandas_rule,
    get_frequency_timedelta,
    get_simu_source,
    get_source_aggregation_method,
    get_source_native_frequency,
    normalize_bc_tag,
)
from iriscc.datautils import Data, crop_domain_from_ds, interpolation_target_grid, reformat_as_target


def dataset_variant_dir(exp: str, variant: str) -> Path:
    return DATASET_BC_DIR / f"dataset_{exp}_test_{variant}"


def source_period_date(exp: str, period: str):
    if period == "train_hist":
        return get_bc_train_hist_dates(exp)[0]
    if period == "test_hist":
        return get_bc_test_hist_dates(exp)[0]
    if period == "test_future":
        return get_bc_test_future_dates(exp)[0]
    raise ValueError(f"Unsupported period: {period}")


def current_frequency_minutes(frequency: str) -> int:
    return int(get_frequency_timedelta(frequency).total_seconds() // 60)


def grouped_dates_for_source(get_data: Data, source_name: str, var: str, dates, ssp: str):
    groups = []
    current_file = None
    current_dates = []
    for date in dates:
        resolved = tuple(get_data._resolve_source_files(source_name, var, date=date, ssp=ssp))
        if resolved != current_file and current_dates:
            groups.append((current_file, current_dates))
            current_dates = []
        current_file = resolved
        current_dates.append(date)
    if current_dates:
        groups.append((current_file, current_dates))
    return groups


def simu_netcdf_path(
    get_data: Data,
    exp: str,
    simu: str,
    var: str,
    ssp: str,
    corrected: bool,
    period: str,
    bc_tag: str | None = None,
) -> Path:
    if corrected:
        return get_bias_corrected_netcdf_path(exp, simu, var, period, ssp=ssp, bc_tag=bc_tag)
    source_name = get_simu_source(exp, simu)
    return Path(get_data._resolve_source_file(source_name, var, date=source_period_date(exp, period), ssp=ssp))


def select_date(ds: xr.Dataset, date) -> xr.Dataset:
    matches = ds.sel(time=ds.time == pd.Timestamp(date))
    if matches.sizes.get("time", 0) == 0:
        normalized = pd.Timestamp(date).normalize()
        ds_times = pd.DatetimeIndex(ds.time.values)
        keep = ds_times.normalize() == normalized
        if int(np.sum(keep)) == 0:
            raise ValueError(f"Missing exact timestamp {pd.Timestamp(date)} in prepared batch.")
        matches = ds.isel(time=keep)
    return matches.isel(time=0, drop=True)


def filter_dates(dates, startdate: str | None, enddate: str | None):
    if startdate is None and enddate is None:
        return dates
    start = pd.to_datetime(startdate) if startdate else dates[0]
    end = pd.to_datetime(enddate) if enddate else dates[-1]
    return [date for date in dates if start <= date <= end]


def resample_batch_frequency(
    ds: xr.Dataset,
    *,
    label: str,
    current_frequency: str,
    target_frequency: str,
    aggregation_method: str,
) -> xr.Dataset:
    if current_frequency == target_frequency:
        return ds
    current_minutes = current_frequency_minutes(current_frequency)
    target_minutes = current_frequency_minutes(target_frequency)
    if target_minutes < current_minutes:
        raise ValueError(
            f"{label} currently available at '{current_frequency}' cannot be expanded to finer "
            f"prediction frequency '{target_frequency}'."
        )
    resampler = ds.resample(time=get_frequency_pandas_rule(target_frequency))
    if aggregation_method == "mean":
        return resampler.mean()
    if aggregation_method == "sum":
        return resampler.sum()
    raise ValueError(f"Unsupported aggregation method '{aggregation_method}' for {label}.")


def select_batch_window(ds: xr.Dataset, dates, *, current_frequency: str, target_frequency: str) -> xr.Dataset:
    start = pd.Timestamp(dates[0])
    if current_frequency == target_frequency == "daily":
        end = pd.Timestamp(dates[-1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        end = pd.Timestamp(dates[-1]) + get_frequency_timedelta(target_frequency) - get_frequency_timedelta(current_frequency)
    return ds.sel(time=slice(start, end))


def prepare_native_target_batches(
    get_data: Data,
    source_name: str,
    var: str,
    batch_dates,
    ssp: str,
    target_file,
    domain,
    target_method: str,
    prediction_frequency: str,
) -> dict[pd.Timestamp, np.ndarray]:
    prepared: dict[pd.Timestamp, np.ndarray] = {}
    target_groups = grouped_dates_for_source(get_data, source_name, var, batch_dates, ssp)
    native_frequency = get_source_native_frequency(source_name)
    aggregation_method = get_source_aggregation_method(source_name, prediction_frequency)

    for target_files, target_dates in target_groups:
        datasets = []
        for path in target_files:
            ds_i = xr.open_dataset(path)
            ds_i = get_data._standardize_source_geometry(
                ds_i,
                get_data.get_source_spec(source_name).get("geometry", "none"),
            )
            datasets.append(ds_i)
        ds_target = xr.concat(datasets, dim="time").sortby("time")
        try:
            ds_target = select_batch_window(
                ds_target,
                target_dates,
                current_frequency=native_frequency,
                target_frequency=prediction_frequency,
            )
            ds_target = resample_batch_frequency(
                ds_target,
                label=f"target source '{source_name}'",
                current_frequency=native_frequency,
                target_frequency=prediction_frequency,
                aggregation_method=aggregation_method,
            )
            ds_target = ds_target.sel(time=pd.DatetimeIndex(target_dates))
            ds_target[var].values = get_data.clean_data(
                ds_target[var].values,
                var,
                data_type=get_data.get_source_spec(source_name).get("data_type"),
            )
            ds_target = reformat_as_target(
                ds_target,
                target_file=target_file,
                domain=domain,
                method=target_method,
                mask=True,
                input_projection=projection_for_source(source_name),
                reuse_weights=True,
            )
            for timestamp in pd.DatetimeIndex(target_dates):
                prepared[pd.Timestamp(timestamp)] = select_date(ds_target, timestamp)[var].values.astype(np.float32)
        finally:
            for ds_i in datasets:
                ds_i.close()
            ds_target.close()

    return prepared


def projection_for_source(source_name: str):
    return ALADIN_PROJ_PYPROJ if source_name == "rcm_aladin" else None


def open_grouped_source_batch(
    get_data: Data,
    source_name: str,
    batch_file,
    *,
    var: str,
    date,
    ssp: str,
) -> xr.Dataset:
    if isinstance(batch_file, (tuple, list)):
        datasets = []
        for path in batch_file:
            ds_i = xr.open_dataset(path)
            ds_i = get_data._standardize_source_geometry(
                ds_i,
                get_data.get_source_spec(source_name).get("geometry", "none"),
            )
            datasets.append(ds_i)
        try:
            return xr.concat(datasets, dim="time").sortby("time")
        finally:
            for ds_i in datasets:
                ds_i.close()
    ds = xr.open_dataset(batch_file)
    return ds


def build_perfect_model_target(
    get_data: Data,
    source_name: str,
    var: str,
    date,
    ssp: str,
    target_file,
    domain,
    method: str,
) -> np.ndarray:
    ds_target_native = get_data.get_rcm_dataset_from_source(source_name, var, date, ssp=ssp)
    try:
        ds_target = reformat_as_target(
            ds_target_native,
            target_file=target_file,
            domain=domain,
            method=method,
            mask=True,
            input_projection=projection_for_source(source_name),
            reuse_weights=True,
        )
        return ds_target[var].values
    finally:
        ds_target_native.close()


def build_coarse_bridge_target(get_data: Data, source_name: str, var: str, date, ssp: str, domain) -> xr.Dataset:
    spec = get_data.get_source_spec(source_name)
    ds = get_data._open_source_dataset(source_name, var, date=date, ssp=ssp)
    try:
        ds = select_date(ds, date)
        if "time" in ds.coords and "time" not in ds.dims:
            ds = ds.drop_vars("time")
        ds = get_data._standardize_source_geometry(ds, spec.get("geometry", "none"))
        if spec.get("geometry") != "rcm":
            ds = crop_domain_from_ds(ds, domain)
        ds[var].values = get_data.clean_data(ds[var].values, var, data_type=spec.get("data_type"))
        return ds
    except Exception:
        ds.close()
        raise


def prepare_input_batch(
    get_data: Data,
    ds_batch: xr.Dataset,
    source_name: str,
    var: str,
    batch_dates,
    ssp: str,
    target_file,
    domain,
    bc_domain,
    perfect_model_input_grid_source: str | None,
    coarse_method: str,
    target_method: str,
) -> xr.Dataset:
    """Return input fields on the ML target grid for a whole source-file batch."""
    if perfect_model_input_grid_source is None:
        return reformat_as_target(
            ds_batch,
            target_file=target_file,
            domain=domain,
            method=target_method,
            mask=True,
            input_projection=projection_for_source(source_name),
            reuse_weights=True,
        )

    coarse_target = build_coarse_bridge_target(
        get_data,
        perfect_model_input_grid_source,
        var,
        batch_dates[0],
        ssp,
        bc_domain,
    )
    try:
        ds_coarse = interpolation_target_grid(
            ds_batch,
            ds_target=coarse_target,
            method=coarse_method,
            input_projection=projection_for_source(source_name),
            target_projection=projection_for_source(perfect_model_input_grid_source),
            reuse_weights=True,
        )
        if "time" in ds_coarse.coords and "time" not in ds_coarse.dims:
            ds_coarse = ds_coarse.drop_vars("time")
    finally:
        coarse_target.close()

    return reformat_as_target(
        ds_coarse,
        target_file=target_file,
        domain=domain,
        method=target_method,
        mask=True,
        input_projection=projection_for_source(perfect_model_input_grid_source),
        reuse_weights=True,
    )


def prepare_bc_batch(
    *,
    exp: str,
    simu: str,
    var: str,
    period: str,
    ssp: str,
    bc_tag: str | None,
    startdate,
    enddate,
    target_file,
    domain,
) -> xr.Dataset:
    bc_path = get_bias_corrected_netcdf_path(exp, simu, var, period, ssp=ssp, bc_tag=bc_tag)
    ds_bc = xr.open_dataset(bc_path)
    if "time" in ds_bc.coords:
        ds_bc = ds_bc.sel(time=slice(startdate, enddate))
    return reformat_as_target(
        ds_bc,
        target_file=target_file,
        domain=domain,
        method=CONFIG[exp].get("perfect_model_target_method", "conservative_normed"),
        mask=True,
        input_projection=None,
        reuse_weights=True,
    )


if __name__ == "__main__":
    start_time = utc_now_iso()
    parser = argparse.ArgumentParser(description="Build daily sample datasets from simulation fields")
    parser.add_argument("--exp", type=str, help="Experiment name (e.g., exp5)", default="exp5")
    parser.add_argument("--var", type=str, help="Variable to use", default="tas")
    parser.add_argument("--simu", type=str, help="Simulation family", default="gcm")
    parser.add_argument("--ssp", type=str, default=None, help="Scenario override. Defaults to experiment config.")
    parser.add_argument("--corrected", action="store_true", help="Use bias-corrected simulation files and write dataset_<exp>_test_<simu>_bc")
    parser.add_argument("--bc-tag", type=str, default=None, help="Optional suffix selecting a tagged BC NetCDF family, e.g. sbck_cdft.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional explicit output dataset directory")
    parser.add_argument("--include-train-hist", action="store_true", help="Also package the 1980-1999 historical train window. Useful for standalone perfect-model training datasets.")
    parser.add_argument("--historical-only", action="store_true", help="Skip future-period packaging and keep only historical periods.")
    parser.add_argument("--startdate", type=str, default=None, help="Optional inclusive first date to package, formatted YYYYMMDD.")
    parser.add_argument("--enddate", type=str, default=None, help="Optional inclusive last date to package, formatted YYYYMMDD.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples that already exist in the output directory.")
    parser.add_argument(
        "--conditioning-bc-tag",
        type=str,
        default=None,
        help="Optional BC tag to use for the perfect-model conditioning field, e.g. sbck_cdft.",
    )
    parser.add_argument(
        "--perfect-model-target-source",
        default=None,
        help="Optional native model source used as pseudo-truth y for all packaged dates, e.g. rcm_aladin.",
    )
    parser.add_argument(
        "--perfect-model-target-method",
        default=None,
        help="Regridding method for perfect-model pseudo-truth. Defaults to config or conservative_normed.",
    )
    parser.add_argument("--test", action="store_true", help="Only build the first day from each period")
    args = parser.parse_args()

    exp = args.exp
    var = args.var
    ssp = args.ssp or CONFIG[exp]["ssp"]
    bc_tag = normalize_bc_tag(args.bc_tag)
    domain = CONFIG[exp]["domain"]
    orog_file = CONFIG[exp]["orog_file"]
    target_file = CONFIG[exp]["target_file"]
    perfect_model_target_source = args.perfect_model_target_source or CONFIG[exp].get("perfect_model_target_source")
    perfect_model_condition_on_bc = bool(CONFIG[exp].get("perfect_model_condition_on_bc", False))
    conditioning_bc_tag = normalize_bc_tag(
        args.conditioning_bc_tag
        if args.conditioning_bc_tag is not None
        else CONFIG[exp].get("perfect_model_conditioning_bc_tag")
    )
    perfect_model_input_grid_source = CONFIG[exp].get("perfect_model_input_grid_source")
    perfect_model_input_coarse_method = CONFIG[exp].get("perfect_model_input_coarse_method", "conservative_normed")
    perfect_model_input_target_method = CONFIG[exp].get("perfect_model_input_target_method", "bilinear")
    perfect_model_target_method = (
        args.perfect_model_target_method
        or CONFIG[exp].get("perfect_model_target_method")
        or "conservative_normed"
    )
    variant = f"{args.simu}_bc" if args.corrected else args.simu
    output_dir = Path(args.output_dir) if args.output_dir else dataset_variant_dir(exp, variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_frequency = get_experiment_training_frequency(exp)
    prediction_frequency = get_experiment_prediction_frequency(exp)

    resolved_settings = {
        "exp": exp,
        "var": var,
        "simu": args.simu,
        "ssp": ssp,
        "output_dir": output_dir,
        "training_frequency": training_frequency,
        "prediction_frequency": prediction_frequency,
        "perfect_model_target_source": perfect_model_target_source,
        "perfect_model_condition_on_bc": perfect_model_condition_on_bc,
        "conditioning_bc_tag": conditioning_bc_tag,
        "perfect_model_input_grid_source": perfect_model_input_grid_source,
        "perfect_model_input_coarse_method": perfect_model_input_coarse_method,
        "perfect_model_input_target_method": perfect_model_input_target_method,
        "perfect_model_target_method": perfect_model_target_method,
    }
    path_inventory = inventory_paths(
        {
            "output_dir": output_dir,
            "orog_file": orog_file,
            "target_file": target_file,
        }
    )
    print_resolved_context(
        script_name="build_dataset_pp.py",
        parameters=vars(args),
        settings={**resolved_settings, "path_inventory": path_inventory},
        inputs={
            "orog_file": orog_file,
            "target_file": target_file,
        },
        outputs={"output_dir": output_dir},
    )

    get_data = Data(domain=domain)
    source_name = get_simu_source(exp, args.simu)
    train_hist_window = get_bc_train_hist_dates(exp)
    test_hist_window = get_bc_test_hist_dates(exp)
    test_future_window = get_bc_test_future_dates(exp)
    dates_bc_train_hist = build_time_range(train_hist_window[0], train_hist_window[-1], prediction_frequency)
    dates_bc_test_hist = build_time_range(test_hist_window[0], test_hist_window[-1], prediction_frequency)
    dates_bc_test_future = build_time_range(test_future_window[0], test_future_window[-1], prediction_frequency)
    ds_orog = xr.open_dataset(orog_file)
    orog = ds_orog["elevation"].values if "elevation" in ds_orog else ds_orog["z"].values
    ds_orog.close()

    periods = []
    if args.include_train_hist:
        periods.append(("train_hist", dates_bc_train_hist))
    periods.append(("test_hist", dates_bc_test_hist))
    if not args.historical_only:
        periods.append(("test_future", dates_bc_test_future))

    for period, dates in periods:
        dates = filter_dates(list(dates), args.startdate, args.enddate)
        if args.test:
            dates = dates[:1]
        if len(dates) == 0:
            continue
        if args.corrected:
            date_batches = [(
                simu_netcdf_path(get_data, exp, args.simu, var, ssp, args.corrected, period, bc_tag=bc_tag),
                list(dates),
            )]
        else:
            date_batches = grouped_dates_for_source(get_data, source_name, var, dates, ssp)
        for batch_file, batch_dates in date_batches:
            ds_batch = open_grouped_source_batch(
                get_data,
                source_name,
                batch_file,
                var=var,
                date=batch_dates[0],
                ssp=ssp,
            )
            ds_input_batch = None
            ds_bc_batch = None
            try:
                if not args.corrected:
                    ds_batch = get_data._standardize_source_geometry(
                        ds_batch,
                        get_data.get_source_spec(source_name).get("geometry", "none"),
                    )
                    ds_batch = select_batch_window(
                        ds_batch,
                        batch_dates,
                        current_frequency=get_source_native_frequency(source_name),
                        target_frequency=prediction_frequency,
                    )
                    ds_batch = resample_batch_frequency(
                        ds_batch,
                        label=f"source '{source_name}'",
                        current_frequency=get_source_native_frequency(source_name),
                        target_frequency=prediction_frequency,
                        aggregation_method=get_source_aggregation_method(source_name, prediction_frequency),
                    )
                    if perfect_model_target_source:
                        ds_input_batch = prepare_input_batch(
                            get_data,
                            ds_batch,
                            source_name,
                            var,
                            batch_dates,
                            ssp,
                            target_file,
                            domain,
                            CONFIG[exp].get("bc_domain", domain),
                            perfect_model_input_grid_source,
                            perfect_model_input_coarse_method,
                            perfect_model_input_target_method,
                        )
                    else:
                        ds_input_batch = None
                    target_map = {}
                    if perfect_model_target_source:
                        target_map = prepare_native_target_batches(
                            get_data,
                            perfect_model_target_source,
                            var,
                            batch_dates,
                            ssp,
                            target_file,
                            domain,
                            perfect_model_target_method,
                            prediction_frequency,
                        )
                    elif batch_dates[-1] <= dates_bc_test_hist[-1]:
                        target_map = prepare_native_target_batches(
                            get_data,
                            CONFIG[exp].get("target_source"),
                            var,
                            batch_dates,
                            ssp,
                            target_file,
                            domain,
                            CONFIG[exp].get("perfect_model_target_method", "conservative_normed"),
                            prediction_frequency,
                        )
                    if perfect_model_target_source and perfect_model_condition_on_bc:
                        ds_bc_batch = prepare_bc_batch(
                            exp=exp,
                            simu=args.simu,
                            var=var,
                            period=period,
                            ssp=ssp,
                            bc_tag=conditioning_bc_tag,
                            startdate=batch_dates[0],
                            enddate=batch_dates[-1],
                            target_file=target_file,
                            domain=domain,
                        )
                        ds_bc_batch = resample_batch_frequency(
                            ds_bc_batch,
                            label=f"bias-corrected source '{args.simu}'",
                            current_frequency=training_frequency,
                            target_frequency=prediction_frequency,
                            aggregation_method=get_source_aggregation_method(source_name, prediction_frequency),
                        )
                else:
                    ds_input_batch = None
                    target_map = {}
                    ds_batch = select_batch_window(
                        ds_batch,
                        batch_dates,
                        current_frequency=training_frequency,
                        target_frequency=prediction_frequency,
                    )
                    ds_batch = resample_batch_frequency(
                        ds_batch,
                        label=f"bias-corrected source '{args.simu}'",
                        current_frequency=training_frequency,
                        target_frequency=prediction_frequency,
                        aggregation_method=get_source_aggregation_method(source_name, prediction_frequency),
                    )
                    if batch_dates[-1] <= dates_bc_test_hist[-1]:
                        target_map = prepare_native_target_batches(
                            get_data,
                            CONFIG[exp].get("target_source"),
                            var,
                            batch_dates,
                            ssp,
                            target_file,
                            domain,
                            CONFIG[exp].get("perfect_model_target_method", "conservative_normed"),
                            prediction_frequency,
                        )
                for date in batch_dates:
                    sample_path = output_dir / f"sample_{format_sample_time_token(date, prediction_frequency)}.npz"
                    if args.skip_existing and sample_path.exists():
                        print(f"{date} exists")
                        continue
                    print(date)
                    if ds_input_batch is not None:
                        ds_i = select_date(ds_input_batch, date)
                    else:
                        ds_i = select_date(ds_batch, date)
                        ds_i = reformat_as_target(
                            ds_i,
                            target_file=target_file,
                            domain=domain,
                            method=perfect_model_input_target_method,
                            mask=True,
                            input_projection=projection_for_source(source_name) if not args.corrected else None,
                            reuse_weights=True,
                        )

                    x_fields = [orog, ds_i[var].values]
                    if ds_bc_batch is not None:
                        ds_bc_i = select_date(ds_bc_batch, date)
                        x_fields.append(ds_bc_i[var].values)
                    x = np.stack(x_fields, axis=0)
                    sample = {"x": x.astype(np.float32)}

                    if perfect_model_target_source:
                        y = target_map[pd.Timestamp(date)]
                        sample["y"] = np.expand_dims(y, axis=0)
                    elif date <= dates_bc_test_hist[-1]:
                        y = target_map[pd.Timestamp(date)]
                        sample["y"] = np.expand_dims(y, axis=0)

                    np.savez(sample_path, **sample)
            finally:
                if ds_bc_batch is not None:
                    ds_bc_batch.close()
                if ds_input_batch is not None:
                    ds_input_batch.close()
                ds_batch.close()
    prov_path = write_provjson(
        output_dir / "provenance_build_dataset.prov.json",
        build_prov_bundle(
            script_name="build_dataset_pp.py",
            activity_type="dataset_build",
            start_time=start_time,
            end_time=utc_now_iso(),
            parameters=vars(args),
            settings={**resolved_settings, "path_inventory": path_inventory},
            inputs={
                "orog_file": orog_file,
                "target_file": target_file,
            },
            outputs={"output_dir": output_dir},
            cwd=Path(__file__).resolve().parents[2],
        ),
    )
    print(f"provenance_provjson={prov_path}", flush=True)
