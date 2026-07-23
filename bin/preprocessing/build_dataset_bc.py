import sys
sys.path.append(".")

from pathlib import Path
import os
import numpy as np
import pandas as pd
import argparse

from bin.preprocessing.build_dataset import Data
from iriscc.datautils import (interpolation_target_grid,
                               crop_domain_from_ds)
from iriscc.settings import (ALADIN_PROJ_PYPROJ,
                             DATASET_BC_DIR,
                             CONFIG,
                             get_bc_bundle_path,
                             get_bc_test_future_dates,
                             get_bc_test_hist_dates,
                             get_bc_train_hist_dates,
                             get_frequency_pandas_rule,
                             get_experiment_training_frequency,
                             get_simu_family,
                             get_simu_source,
                             get_source_aggregation_method,
                             get_source_default_frequency,
                             get_source_native_frequency)

ERA5_BC_DOMAIN_MARGIN = float(os.getenv("IDOWNSCALE_ERA5_BC_DOMAIN_MARGIN", "0.0"))


def require_expected_time_length(ds, dates, source_name: str, var: str, label: str):
    expected = len(pd.DatetimeIndex(dates))
    if "time" not in ds.dims:
        raise ValueError(
            f"BC {label} source '{source_name}' for variable '{var}' has no time dimension. "
            "Check that the source pattern resolves to a time-varying data file, not a static field."
        )
    actual = int(ds.sizes.get("time", 0))
    if actual != expected:
        raise ValueError(
            f"BC {label} source '{source_name}' for variable '{var}' has {actual} time steps, "
            f"but {expected} were requested ({pd.DatetimeIndex(dates)[0]} -> {pd.DatetimeIndex(dates)[-1]})."
        )
    return ds


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Isolated Decoupled BC Dataset Builder")
    parser.add_argument("--simu", type=str, default="simu")
    parser.add_argument("--ssp", type=str, default="historical")
    parser.add_argument("--var", type=str, default="tas")
    parser.add_argument("--paired-vars", type=str, default=None, help="Optional comma-separated pair of variables to package jointly, e.g. uas,vas.")
    parser.add_argument("--exp", type=str, default="exp5")
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test", action="store_true", help="Test mode: process 1 day per period")
    args = parser.parse_args()
    paired_vars = [item.strip() for item in (args.paired_vars or "").split(",") if item.strip()]
    if paired_vars and len(paired_vars) != 2:
        raise ValueError("--paired-vars requires exactly two comma-separated variables.")
    variables = paired_vars or [args.var]
    bundle_var_tag = variables if paired_vars else None

    domain = CONFIG[args.exp]["domain"]
    bc_domain = CONFIG[args.exp].get("bc_domain", domain)
    bc_reanalysis_source = CONFIG[args.exp].get("bc_reanalysis_source", "era5")
    gcm_source = CONFIG[args.exp].get("gcm_source", "gcm_cnrm_cm6_1")
    simu_source = get_simu_source(args.exp, args.simu)
    simu_family = get_simu_family(args.exp, args.simu)
    target_regrid_method = CONFIG[args.exp].get("perfect_model_target_method", "conservative_normed")
    get_bc_data = Data(domain=bc_domain)
    dates_bc_train_hist = get_bc_train_hist_dates(args.exp)
    dates_bc_test_hist = get_bc_test_hist_dates(args.exp)
    dates_bc_test_future = get_bc_test_future_dates(args.exp)

    base_dir = Path(args.output_dir) if args.output_dir else DATASET_BC_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    def grouped_dates_for_source(source_name, dates):
        groups = []
        current_file = None
        current_dates = []
        for date in pd.DatetimeIndex(dates):
            resolved = get_bc_data._resolve_source_file(source_name, variables[0], date=date, ssp=args.ssp)
            if resolved != current_file and current_dates:
                groups.append((current_file, pd.DatetimeIndex(current_dates)))
                current_dates = []
            current_file = resolved
            current_dates.append(date)
        if current_dates:
            groups.append((current_file, pd.DatetimeIndex(current_dates)))
        return groups

    workflow_frequency = get_experiment_training_frequency(args.exp)

    def select_frequency_window(ds, dates, source_name):
        dates = pd.DatetimeIndex(dates).sort_values()
        if "time" not in ds.dims:
            raise ValueError(
                f"Source '{source_name}' has no time dimension for requested window {dates[0]} -> {dates[-1]}."
            )
        source_frequency = get_source_native_frequency(source_name)
        expected_frequency = get_source_default_frequency(source_name)
        if expected_frequency != workflow_frequency:
            raise ValueError(
                f"Workflow output frequency '{workflow_frequency}' does not match the configured default "
                f"frequency '{expected_frequency}' for source '{source_name}'."
            )
        stop = dates[-1] + pd.Timedelta(get_frequency_pandas_rule(workflow_frequency)) - pd.Timedelta(seconds=1)
        ds = ds.sel(time=slice(dates[0], stop))
        if ds.sizes.get("time", 0) == 0:
            raise ValueError(f"No time values found for requested window {dates[0]} -> {dates[-1]}")
        if source_frequency != workflow_frequency:
            aggregation_method = get_source_aggregation_method(source_name, workflow_frequency)
            resampler = ds.resample(time=get_frequency_pandas_rule(workflow_frequency))
            if aggregation_method == "mean":
                ds = resampler.mean()
            elif aggregation_method == "sum":
                ds = resampler.sum()
            else:
                raise ValueError(
                    f"Unsupported aggregation method '{aggregation_method}' for source '{source_name}'."
                )
        ds_times = pd.DatetimeIndex(ds.time.values)
        if workflow_frequency == "daily":
            common = pd.Index(dates.normalize()).intersection(pd.Index(ds_times.normalize()))
            if len(common) == 0:
                raise ValueError(
                    f"No overlapping {workflow_frequency} values found for requested window {dates[0]} -> {dates[-1]}"
                )
            keep = ds_times.normalize().isin(common)
            ds = ds.isel(time=keep)
            ds = ds.assign_coords(time=("time", common.values))
            return ds
        common = pd.Index(dates).intersection(pd.Index(ds_times))
        if len(common) == 0:
            raise ValueError(
                f"No overlapping {workflow_frequency} values found for requested window {dates[0]} -> {dates[-1]}"
            )
        ds = ds.sel(time=common)
        return ds

    def load_batch_dataset(source_name, dates, var, *, domain_override=None):
        spec = get_bc_data.get_source_spec(source_name)
        ds = get_bc_data._open_source_dataset(source_name, var, date=dates[0], ssp=args.ssp)
        ds = select_frequency_window(ds, dates, source_name)
        ds = require_expected_time_length(ds, dates, source_name, var, "dataset")
        if spec.get("geometry") != "rcm":
            ds = crop_domain_from_ds(ds, domain_override if domain_override is not None else get_bc_data.domain)
        ds[var].values = get_bc_data.clean_data(ds[var].values, var, data_type=spec.get("data_type"))
        return ds

    def process_period(dates, label):
        print(f"--- Processing {label} Period ---", flush=True)
        if args.test:
            dates = dates[:1]
            print(f"TEST_MODE: Limiting to {dates[0].date()}", flush=True)

        era5_list = []
        simu_list = []
        date_batches = grouped_dates_for_source(simu_source, dates)

        # Generic Geometry Anchor discovery for Future periods
        is_future = (label == "test_future")
        era5_domain = [
            bc_domain[0] - ERA5_BC_DOMAIN_MARGIN,
            bc_domain[1] + ERA5_BC_DOMAIN_MARGIN,
            bc_domain[2] - ERA5_BC_DOMAIN_MARGIN,
            bc_domain[3] + ERA5_BC_DOMAIN_MARGIN,
        ]
        if simu_family == "gcm":
            target_grid = load_batch_dataset(simu_source, pd.DatetimeIndex([dates[0]]), variables[0]).isel(time=0, drop=True)
        else:
            ds_gcm_target = load_batch_dataset(gcm_source, pd.DatetimeIndex([dates[0]]), variables[0])
            # Preserve the archival BC geometry on the GCM bridge grid; the
            # later sample materialization step is responsible for reformatting
            # corrected fields onto the final target domain.
            target_grid = ds_gcm_target.isel(time=0, drop=True)

        for batch_file, batch_dates in date_batches:
            print(
                f"[{label}] batch {batch_dates[0].date()} -> {batch_dates[-1].date()} from {Path(batch_file).name}",
                flush=True,
            )
            simu_vars = []
            for var in variables:
                ds_simu = load_batch_dataset(simu_source, batch_dates, var)

                if simu_family == "rcm":
                    input_projection = ALADIN_PROJ_PYPROJ if simu_source == "rcm_aladin" else None
                    ds_simu = interpolation_target_grid(
                        ds_simu,
                        ds_target=target_grid,
                        method=target_regrid_method,
                        input_projection=input_projection,
                    )
                simu_vars.append(ds_simu[var].values)

            if not is_future:
                era5_vars = []
                for var in variables:
                    era5_batch_list = []
                    for _, era5_batch_dates in grouped_dates_for_source(bc_reanalysis_source, batch_dates):
                        ds_era5 = load_batch_dataset(
                            bc_reanalysis_source,
                            era5_batch_dates,
                            var,
                            domain_override=era5_domain,
                        )
                        ds_target_regrid = interpolation_target_grid(
                            ds_era5,
                            ds_target=target_grid,
                            method=target_regrid_method,
                        )
                        era5_batch_list.append(ds_target_regrid[var].values)
                    era5_vars.append(np.concatenate(era5_batch_list, axis=0))
                era5_list.append(np.stack(era5_vars, axis=-1) if paired_vars else era5_vars[0])

            simu_list.append(np.stack(simu_vars, axis=-1) if paired_vars else simu_vars[0])

        # Persistence Logic (Isolated Schema)
        simu_stack = np.concatenate(simu_list, axis=0)
        output_path = (
            Path(args.output_dir) / (
                f"bc_{label}_{args.simu}_{'_'.join(variables)}.npz" if paired_vars else f"bc_{label}_{args.simu}.npz"
            )
            if args.output_dir
            else get_bc_bundle_path(args.exp, args.simu, label, variables=bundle_var_tag)
        )

        save_dict = {args.simu: simu_stack, "dates": dates}
        if not is_future:
            save_dict["era5"] = np.concatenate(era5_list, axis=0)
        if paired_vars:
            save_dict["variables"] = np.asarray(variables, dtype=str)

        np.savez_compressed(output_path, **save_dict)
        print(f"--- {label} Isolated Volume Saved to {output_path} ---", flush=True)
        return simu_stack

    # Apply CLI Overrides for 31-Day Certification Benchmarking
    if args.start_date and args.end_date:
        print(f"--- Applying Benchmark Date Overrides: {args.start_date} to {args.end_date} ---", flush=True)
        dates_bc_train_hist = pd.date_range(
            start=args.start_date,
            end=args.end_date,
            freq=get_frequency_pandas_rule(workflow_frequency),
        )
        dates_bc_test_hist = pd.Index([])
        dates_bc_test_future = pd.Index([])

    # 1. Historical Train (with reanalysis)
    if len(dates_bc_train_hist) > 0:
        process_period(dates_bc_train_hist, "train_hist")

    # 2. Historical Test (with reanalysis)
    if len(dates_bc_test_hist) > 0:
        process_period(dates_bc_test_hist, "test_hist")

    # 3. Future Projection (decoupled, no reanalysis)
    if len(dates_bc_test_future) > 0:
        process_period(dates_bc_test_future, "test_future")

    print("--- ISOLATED PRODUCTION SYNTHESIS COMPLETE ---", flush=True)
