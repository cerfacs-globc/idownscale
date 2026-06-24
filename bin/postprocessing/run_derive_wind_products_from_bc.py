#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iriscc.settings import CONFIG, get_bias_corrected_netcdf_path, normalize_bc_tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive wind products from bias-corrected component NetCDF files.")
    parser.add_argument("--exp", required=True)
    parser.add_argument("--simu", default="gcm")
    parser.add_argument("--ssp", default=None)
    parser.add_argument("--u-var", default="uas")
    parser.add_argument("--v-var", default="vas")
    parser.add_argument("--speed-var", default="sfcWind")
    parser.add_argument("--direction-var", default="windFromDirection")
    parser.add_argument("--bc-tag", default="sbck_mbcn")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--direction", action="store_true", help="Also derive wind-from direction.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(f"[derive] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    ssp = args.ssp or CONFIG[args.exp]["ssp"]
    bc_tag = normalize_bc_tag(args.bc_tag)
    for period in ("train_hist", "test_hist", "test_future"):
        u_path = get_bias_corrected_netcdf_path(args.exp, args.simu, args.u_var, period, ssp=ssp, bc_tag=bc_tag)
        v_path = get_bias_corrected_netcdf_path(args.exp, args.simu, args.v_var, period, ssp=ssp, bc_tag=bc_tag)
        speed_output = get_bias_corrected_netcdf_path(args.exp, args.simu, args.speed_var, period, ssp=ssp, bc_tag=bc_tag)
        command = [
            args.python_bin,
            "bin/postprocessing/derive_wind_products.py",
            "--u-path",
            str(u_path),
            "--v-path",
            str(v_path),
            "--u-var",
            args.u_var,
            "--v-var",
            args.v_var,
            "--speed-output",
            str(speed_output),
            "--speed-var",
            args.speed_var,
        ]
        if args.direction:
            direction_output = get_bias_corrected_netcdf_path(
                args.exp, args.simu, args.direction_var, period, ssp=ssp, bc_tag=bc_tag
            )
            command.extend(
                [
                    "--direction-output",
                    str(direction_output),
                    "--direction-var",
                    args.direction_var,
                ]
            )
        run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
