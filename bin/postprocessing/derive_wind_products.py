#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from iriscc.windutils import direction_from_components, speed_from_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive wind speed and direction from paired component NetCDF files.")
    parser.add_argument("--u-path", required=True, help="Path to the zonal component NetCDF file.")
    parser.add_argument("--v-path", required=True, help="Path to the meridional component NetCDF file.")
    parser.add_argument("--u-var", default="uas", help="Variable name in the zonal-component file.")
    parser.add_argument("--v-var", default="vas", help="Variable name in the meridional-component file.")
    parser.add_argument("--speed-output", default=None, help="Optional output NetCDF path for derived wind speed.")
    parser.add_argument("--direction-output", default=None, help="Optional output NetCDF path for derived wind-from direction.")
    parser.add_argument("--speed-var", default="sfcWind", help="Output variable name for derived wind speed.")
    parser.add_argument("--direction-var", default="windFromDirection", help="Output variable name for derived wind-from direction.")
    return parser.parse_args()


def build_output_dataset(template_ds: xr.Dataset, template_var: str, output_var: str, values) -> xr.Dataset:
    coords = {}
    for name, coord in template_ds.coords.items():
        coords[name] = (coord.dims, coord.values) if coord.dims else coord.values
    return xr.Dataset(
        data_vars={
            output_var: (template_ds[template_var].dims, values, dict(template_ds[template_var].attrs)),
        },
        coords=coords,
        attrs=dict(template_ds.attrs),
    )


def validate_component_alignment(ds_u: xr.Dataset, ds_v: xr.Dataset, u_var: str, v_var: str) -> None:
    if ds_u[u_var].dims != ds_v[v_var].dims:
        raise ValueError(f"Component dimensions differ: {ds_u[u_var].dims} vs {ds_v[v_var].dims}")
    for dim in ds_u[u_var].dims:
        if ds_u.sizes[dim] != ds_v.sizes[dim]:
            raise ValueError(f"Component dimension '{dim}' differs: {ds_u.sizes[dim]} vs {ds_v.sizes[dim]}")
    for coord_name in set(ds_u.coords).intersection(ds_v.coords):
        left = ds_u.coords[coord_name]
        right = ds_v.coords[coord_name]
        if left.dims != right.dims:
            raise ValueError(f"Coordinate '{coord_name}' dimensions differ: {left.dims} vs {right.dims}")
        if left.shape != right.shape or not np.array_equal(left.values, right.values, equal_nan=True):
            raise ValueError(f"Coordinate '{coord_name}' values differ between component files.")


def main() -> int:
    args = parse_args()
    if not args.speed_output and not args.direction_output:
        raise ValueError("At least one of --speed-output or --direction-output must be provided.")

    with xr.open_dataset(args.u_path) as ds_u, xr.open_dataset(args.v_path) as ds_v:
        validate_component_alignment(ds_u, ds_v, args.u_var, args.v_var)
        u = ds_u[args.u_var].values
        v = ds_v[args.v_var].values

        if args.speed_output:
            speed_ds = build_output_dataset(ds_u, args.u_var, args.speed_var, speed_from_components(u, v))
            speed_ds[args.speed_var].attrs["units"] = "m/s"
            speed_path = Path(args.speed_output)
            speed_path.parent.mkdir(parents=True, exist_ok=True)
            speed_ds.to_netcdf(speed_path)
            speed_ds.close()

        if args.direction_output:
            direction_ds = build_output_dataset(ds_u, args.u_var, args.direction_var, direction_from_components(u, v))
            direction_ds[args.direction_var].attrs["units"] = "degree"
            direction_path = Path(args.direction_output)
            direction_path.parent.mkdir(parents=True, exist_ok=True)
            direction_ds.to_netcdf(direction_path)
            direction_ds.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
