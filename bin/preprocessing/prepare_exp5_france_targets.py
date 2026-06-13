#!/usr/bin/env python3
"""
Prepare France-focused target files for exp5-like observation targets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

sys.path.append(".")

from iriscc.datautils import (
    crop_domain_from_ds,
    interpolation_target_grid,
    standardize_cerra_geometry,
    standardize_eobs_geometry,
    standardize_era5_geometry,
    standardize_longitudes,
)
from iriscc.settings import (
    CONFIG,
    CERRA_RAW_DIR,
    ERA5_OROG_FILE,
    LANDSEAMASK_EOBS,
    LANDSEAMASK_EOBS_FRANCE,
    OROG_EOBS_EUROPE_FILE,
    OROG_EOBS_FRANCE_FILE,
    TARGET_EOBS_EUROPE_FILE,
    TARGET_EOBS_FRANCE_FILE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare France target files for exp5-like experiments.")
    parser.add_argument("--exp", default="exp5", help="Experiment name.")
    parser.add_argument(
        "--tas-input",
        default=None,
        help="Europe-scale E-OBS temperature file.",
    )
    parser.add_argument(
        "--elevation-input",
        default=None,
        help="Europe-scale E-OBS elevation file.",
    )
    parser.add_argument(
        "--mask-input",
        default=None,
        help="Europe-scale E-OBS land-sea mask file.",
    )
    parser.add_argument(
        "--tas-output",
        default=None,
        help="France-focused E-OBS temperature output.",
    )
    parser.add_argument(
        "--elevation-output",
        default=None,
        help="France-focused E-OBS elevation output.",
    )
    parser.add_argument(
        "--mask-output",
        default=None,
        help="Optional France-focused E-OBS mask output.",
    )
    parser.add_argument(
        "--include-mask",
        action="store_true",
        help="Also generate a France-focused land-sea mask subset for convenience.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing France-focused files. By default, existing outputs are kept.",
    )
    return parser.parse_args()


def prepare_subset(input_path: Path, output_path: Path, domain: list[float], force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"Keeping existing file: {output_path}")
        return
    ds = xr.open_dataset(input_path, engine="netcdf4")
    ds = standardize_eobs_geometry(ds)
    ds = crop_domain_from_ds(ds, domain)
    # The historical France files in the repo expose only lon/lat coordinates.
    ds = ds.drop_vars(["x", "y"], errors="ignore")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)


def prepare_cerra_reference(output_path: Path, domain: list[float], force: bool = False) -> xr.Dataset:
    if output_path.exists() and not force:
        print(f"Keeping existing file: {output_path}")
        return xr.open_dataset(output_path, engine="netcdf4")

    files = sorted((CERRA_RAW_DIR / "tas_3h").glob("tas_3h_CERRA_1984_*.nc"))
    if not files:
        raise FileNotFoundError(f"Missing CERRA tas files under {CERRA_RAW_DIR / 'tas_3h'}")

    datasets = []
    for path in files:
        ds = xr.open_dataset(path, engine="netcdf4")
        ds = standardize_cerra_geometry(ds)
        ds = ds.isel(time=0)
        datasets.append(ds)
    ds_target = xr.concat(datasets, dim="time").mean("time")
    ds_target = crop_domain_from_ds(ds_target, domain)
    ds_target = ds_target[["tas"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds_target.to_netcdf(output_path)
    return ds_target


def prepare_cerra_elevation(output_path: Path, ds_target: xr.Dataset, force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"Keeping existing file: {output_path}")
        return

    ds_era5_orog = xr.open_dataset(ERA5_OROG_FILE, engine="netcdf4")
    if "time" in ds_era5_orog.dims:
        ds_era5_orog = ds_era5_orog.isel(time=0)
    ds_era5_orog = standardize_era5_geometry(ds_era5_orog)
    ds_era5_orog = standardize_longitudes(ds_era5_orog)
    ds_era5_orog["elevation"] = ds_era5_orog["z"] / 9.80665
    ds_target_orog = interpolation_target_grid(
        ds_era5_orog[["elevation"]],
        ds_target=ds_target,
        method="bilinear",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds_target_orog[["elevation"]].to_netcdf(output_path)


def main() -> int:
    args = parse_args()
    exp_cfg = CONFIG[args.exp]
    domain = exp_cfg["domain"]
    target_source = exp_cfg.get("target_source", exp_cfg["target"])

    if target_source == "cerra":
        tas_input = Path(args.tas_input) if args.tas_input else CERRA_RAW_DIR / "tas_3h"
        elevation_input = Path(args.elevation_input) if args.elevation_input else ERA5_OROG_FILE
        mask_input = Path(args.mask_input) if args.mask_input else LANDSEAMASK_EOBS
        tas_output = Path(args.tas_output) if args.tas_output else Path(exp_cfg["target_file"])
        elevation_output = Path(args.elevation_output) if args.elevation_output else Path(exp_cfg["orog_file"])
        mask_output = Path(args.mask_output) if args.mask_output else LANDSEAMASK_EOBS_FRANCE
    else:
        tas_input = Path(args.tas_input) if args.tas_input else TARGET_EOBS_EUROPE_FILE
        elevation_input = Path(args.elevation_input) if args.elevation_input else OROG_EOBS_EUROPE_FILE
        mask_input = Path(args.mask_input) if args.mask_input else LANDSEAMASK_EOBS
        tas_output = Path(args.tas_output) if args.tas_output else TARGET_EOBS_FRANCE_FILE
        elevation_output = Path(args.elevation_output) if args.elevation_output else OROG_EOBS_FRANCE_FILE
        mask_output = Path(args.mask_output) if args.mask_output else LANDSEAMASK_EOBS_FRANCE

    print(f"Preparing France target files for {args.exp} on domain {domain}")
    if target_source == "cerra":
        ds_target = prepare_cerra_reference(tas_output, domain, force=args.force)
        prepare_cerra_elevation(elevation_output, ds_target, force=args.force)
        ds_target.close()
    else:
        if not tas_input.exists():
            raise FileNotFoundError(f"Missing E-OBS temperature input: {tas_input}")
        if not elevation_input.exists():
            raise FileNotFoundError(f"Missing E-OBS elevation input: {elevation_input}")
        if args.include_mask and not mask_input.exists():
            raise FileNotFoundError(f"Missing E-OBS mask input: {mask_input}")

        prepare_subset(tas_input, tas_output, domain, force=args.force)
        prepare_subset(elevation_input, elevation_output, domain, force=args.force)
        if args.include_mask:
            prepare_subset(mask_input, mask_output, domain, force=args.force)

    print(f"Created: {tas_output}")
    print(f"Created: {elevation_output}")
    if args.include_mask and target_source != "cerra":
        print(f"Created: {mask_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
