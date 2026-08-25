#!/usr/bin/env python3
"""
Build a target-grid elevation file from a higher-resolution topography source.
"""

import argparse
from pathlib import Path

import xarray as xr

from iriscc.datautils import crop_domain_from_ds, interpolation_target_grid, standardize_latlon_geometry, standardize_longitudes
from iriscc.settings import CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a target-grid topography file from a high-resolution source such as ETOPO.")
    parser.add_argument("--exp", required=True, help="Experiment name used to infer domain/target defaults.")
    parser.add_argument("--topography-input", required=True, help="Input topography file, e.g. an ETOPO NetCDF.")
    parser.add_argument("--target-file", default=None, help="Target grid reference file. Defaults to CONFIG[exp]['target_file'].")
    parser.add_argument("--output-file", default=None, help="Output elevation file. Defaults to CONFIG[exp]['orog_file'].")
    parser.add_argument("--topography-var", default=None, help="Optional source variable name. Auto-detected when omitted.")
    parser.add_argument("--method", default="bilinear", choices=["bilinear", "conservative_normed"], help="Regridding method.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    return parser.parse_args()


def detect_topography_var(ds: xr.Dataset, explicit_name=None) -> str:
    if explicit_name:
        if explicit_name not in ds.data_vars:
            raise KeyError(f"Variable '{explicit_name}' not found in topography file. Available: {list(ds.data_vars)}")
        return explicit_name

    for candidate in ["elevation", "z", "topo", "topography", "bedrock"]:
        if candidate in ds.data_vars:
            return candidate
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise KeyError(f"Could not infer topography variable from {list(ds.data_vars)}")


def main() -> int:
    args = parse_args()
    cfg = CONFIG[args.exp]
    target_file = Path(args.target_file) if args.target_file else Path(cfg["target_file"])
    output_file = Path(args.output_file) if args.output_file else Path(cfg["orog_file"])

    if output_file.exists() and not args.force:
        print(f"Keeping existing file: {output_file}")
        return 0

    ds_target = xr.open_dataset(target_file, engine="netcdf4")
    if "time" in ds_target.dims:
        ds_target = ds_target.isel(time=0, drop=True)
    ds_target = standardize_latlon_geometry(ds_target, add_xy_aliases=True)
    ds_target = standardize_longitudes(ds_target)
    ds_target = crop_domain_from_ds(ds_target, cfg["domain"])

    ds_topo = xr.open_dataset(args.topography_input, engine="netcdf4")
    if "time" in ds_topo.dims:
        ds_topo = ds_topo.isel(time=0, drop=True)
    ds_topo = standardize_latlon_geometry(ds_topo, add_xy_aliases=True)
    ds_topo = standardize_longitudes(ds_topo)
    ds_topo = crop_domain_from_ds(ds_topo, cfg["domain"])

    topo_var = detect_topography_var(ds_topo, args.topography_var)
    ds_topo = ds_topo[[topo_var]].rename({topo_var: "elevation"})

    ds_regridded = interpolation_target_grid(ds_topo, ds_target=ds_target, method=args.method, reuse_weights=True)

    if cfg["target_vars"]:
        target_var = cfg["target_vars"][0]
        if target_var in ds_target.data_vars:
            ds_regridded["elevation"] = ds_regridded["elevation"].where(ds_target[target_var].notnull())

    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds_regridded[["elevation"]].to_netcdf(output_file)
    print(f"Wrote {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
