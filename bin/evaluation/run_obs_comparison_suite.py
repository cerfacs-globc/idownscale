#!/usr/bin/env python3
"""
Run the default raw/BC/ML comparison diagnostics for observation-target workflows.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run default raw/BC/ML comparison diagnostics.")
    parser.add_argument("--exp", required=True, help="Experiment name.")
    parser.add_argument("--simu", default="gcm", help="Raw simulation variant, e.g. gcm or rcm.")
    parser.add_argument("--simu-test", default="gcm_bc", help="Bias-corrected simulation variant, e.g. gcm_bc.")
    parser.add_argument("--var", default="tas", help="Variable name.")
    parser.add_argument("--startdate", required=True, help="Metrics start date, YYYYMMDD.")
    parser.add_argument("--enddate", required=True, help="Metrics end date, YYYYMMDD.")
    parser.add_argument("--value-startdate", required=True, help="VALUE start date, YYYYMMDD.")
    parser.add_argument("--value-enddate", required=True, help="VALUE end date, YYYYMMDD.")
    parser.add_argument("--ml-models", default="", help="Comma-separated ML test names.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for subprocess calls.")
    parser.add_argument("--stride-days", type=int, default=7, help="Sampling stride in days for the distribution plot.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print(f"[compare] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    raw_test = f"{args.simu}_raw"
    bc_test = "baseline"

    for test_name, simu_test in ((raw_test, None), (bc_test, args.simu_test)):
        common = [
            "--exp",
            args.exp,
            "--test-name",
            test_name,
            "--startdate",
            args.startdate,
            "--enddate",
            args.enddate,
        ]
        if simu_test:
            common.extend(["--simu-test", simu_test])

        run_command([args.python_bin, "bin/evaluation/compute_test_metrics_day.py", *common])
        run_command([args.python_bin, "bin/evaluation/compute_test_metrics_month.py", *common])
        run_command(
            [
                args.python_bin,
                "bin/evaluation/compute_value_metrics.py",
                "--exp",
                args.exp,
                "--test-name",
                test_name,
                "--simu",
                args.simu,
                "--startdate",
                args.value_startdate,
                "--enddate",
                args.value_enddate,
            ]
            + (["--simu-test", simu_test] if simu_test else [])
        )
        metric_suffix = f"{test_name}_{simu_test}" if simu_test else test_name
        run_command([args.python_bin, "bin/evaluation/plot_test_metrics.py", "--exp", args.exp, "--test-name", metric_suffix, "--scale", "daily"])
        run_command([args.python_bin, "bin/evaluation/plot_test_metrics.py", "--exp", args.exp, "--test-name", metric_suffix, "--scale", "monthly"])

    run_command(
        [
            args.python_bin,
            "bin/evaluation/plot_obs_distribution_comparison.py",
            "--exp",
            args.exp,
            "--simu",
            args.simu,
            "--simu-test",
            args.simu_test,
            "--var",
            args.var,
            "--startdate",
            args.startdate,
            "--enddate",
            args.enddate,
            "--ml-models",
            args.ml_models,
            "--stride-days",
            str(args.stride_days),
        ]
    )


if __name__ == "__main__":
    main()
