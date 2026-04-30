#!/usr/bin/env python3
"""
Prepare France-focused exp5 target files from Europe-scale E-OBS inputs.

This helper makes the pre-Phase-1 target preparation reproducible for users who
start from the original E-OBS Europe-scale products instead of relying on
manually prepared France files already being present in the repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

sys.path.append(".")

from iriscc.datautils import crop_domain_from_ds, standardize_eobs_geometry
from iriscc.settings import (
    CONFIG,
    LANDSEAMASK_EOBS,
    LANDSEAMASK_EOBS_FRANCE,
    OROG_EOBS_EUROPE_FILE,
    OROG_EOBS_FRANCE_FILE,
    TARGET_EOBS_EUROPE_FILE,
    TARGET_EOBS_FRANCE_FILE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare exp5 France target files from Europe-scale E-OBS inputs.")
    parser.add_argument("--exp", default="exp5", help="Experiment name. Currently intended for exp5-like France targets.")
    parser.add_argument(
        "--tas-input",
        default=str(TARGET_EOBS_EUROPE_FILE),
        help="Europe-scale E-OBS temperature file.",
    )
    parser.add_argument(
        "--elevation-input",
        default=str(OROG_EOBS_EUROPE_FILE),
        help="Europe-scale E-OBS elevation file.",
    )
    parser.add_argument(
        "--mask-input",
        default=str(LANDSEAMASK_EOBS),
        help="Europe-scale E-OBS land-sea mask file.",
    )
    parser.add_argument(
        "--tas-output",
        default=str(TARGET_EOBS_FRANCE_FILE),
        help="France-focused E-OBS temperature output.",
    )
    parser.add_argument(
        "--elevation-output",
        default=str(OROG_EOBS_FRANCE_FILE),
        help="France-focused E-OBS elevation output.",
    )
    parser.add_argument(
        "--mask-output",
        default=str(LANDSEAMASK_EOBS_FRANCE),
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


def main() -> int:
    args = parse_args()
    exp_cfg = CONFIG[args.exp]
    domain = exp_cfg["domain"]

    tas_input = Path(args.tas_input)
    elevation_input = Path(args.elevation_input)
    mask_input = Path(args.mask_input)
    tas_output = Path(args.tas_output)
    elevation_output = Path(args.elevation_output)
    mask_output = Path(args.mask_output)

    if not tas_input.exists():
        raise FileNotFoundError(f"Missing E-OBS temperature input: {tas_input}")
    if not elevation_input.exists():
        raise FileNotFoundError(f"Missing E-OBS elevation input: {elevation_input}")
    if args.include_mask and not mask_input.exists():
        raise FileNotFoundError(f"Missing E-OBS mask input: {mask_input}")

    print(f"Preparing France target files for {args.exp} on domain {domain}")
    prepare_subset(tas_input, tas_output, domain, force=args.force)
    prepare_subset(elevation_input, elevation_output, domain, force=args.force)
    if args.include_mask:
        prepare_subset(mask_input, mask_output, domain, force=args.force)

    print(f"Created: {tas_output}")
    print(f"Created: {elevation_output}")
    if args.include_mask:
        print(f"Created: {mask_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
