#!/usr/bin/env python3
"""
Run the exp5 preprocessing workflow with simple step orchestration.

This entrypoint keeps the clean branch usable for day-to-day work:
- central exp5 workflow command
- step-level skip or overwrite behavior
- no hidden shell history required to rebuild the pipeline
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iriscc.settings import CONFIG, DATASET_BC_DIR, GCM_BC_DIR, GRAPHS_DIR, METRICS_DIR, PREDICTION_DIR, RCM_BC_DIR, RUNS_DIR


DEFAULT_STEPS = ["phase1", "stats", "bc_dataset", "bc_apply"]
OPTIONAL_STEPS = [
    "prep_phase1",
    "train",
    "raw_dataset",
    "pp_dataset",
    "predict_loop",
    "metrics_day",
    "metrics_month",
    "value_metrics",
    "plot_metrics_day",
    "plot_metrics_month",
]
ALL_STEPS = DEFAULT_STEPS + OPTIONAL_STEPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exp5 workflow end to end.")
    parser.add_argument("--exp", default="exp5", help="Experiment name. Currently tuned for exp5.")
    parser.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma-separated step list. "
            f"Use 'all' for {','.join(DEFAULT_STEPS)} or include optional steps from {','.join(OPTIONAL_STEPS)}."
        ),
    )
    parser.add_argument(
        "--if-exists",
        choices=["skip", "overwrite"],
        default="skip",
        help="Whether to skip complete outputs or delete and rebuild them.",
    )
    parser.add_argument("--phase1-start-date", default=None, help="Optional YYYYMMDD start date for phase1 dataset generation.")
    parser.add_argument("--phase1-end-date", default=None, help="Optional YYYYMMDD end date for phase1 dataset generation.")
    parser.add_argument("--simu", default="gcm", help="Simulation family for phase2 steps.")
    parser.add_argument("--var", default="tas", help="Variable for phase2 steps.")
    parser.add_argument("--ssp", default=None, help="Override SSP scenario. Defaults to experiment config.")
    parser.add_argument("--test-name", default=None, help="Model run name for inference/evaluation steps.")
    parser.add_argument("--checkpoint-bundle", default=None, help="Optional portable checkpoint bundle directory for inference/evaluation.")
    parser.add_argument("--train-max-epoch", type=int, default=30, help="Max epochs for the optional train step.")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Batch size for the optional train step.")
    parser.add_argument("--train-learning-rate", type=float, default=8e-4, help="Learning rate for the optional train step.")
    parser.add_argument("--train-model", default="unet", help="Model family for the optional train step.")
    parser.add_argument("--train-loss", default=None, help="Optional loss override for the train step.")
    parser.add_argument("--train-output-norm", action="store_true", help="Enable output normalization in the train step.")
    parser.add_argument("--predict-start-date", default="20000101", help="Prediction/evaluation start date.")
    parser.add_argument("--predict-end-date", default="21001231", help="Prediction/evaluation end date.")
    parser.add_argument("--simu-test", default="gcm_bc", help="Inference sample variant, e.g. gcm or gcm_bc.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to use for subprocess steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print decisions without executing commands.")
    return parser.parse_args()


def resolve_steps(raw_steps: str) -> list[str]:
    if raw_steps == "all":
        return list(DEFAULT_STEPS)
    steps = [step.strip() for step in raw_steps.split(",") if step.strip()]
    unknown = [step for step in steps if step not in ALL_STEPS]
    if unknown:
        raise ValueError(f"Unknown steps: {', '.join(unknown)}")
    return steps


def list_phase1_outputs(dataset_dir: Path, start: str | None, end: str | None) -> list[Path]:
    if start and end:
        dates = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq="D")
    else:
        dates = pd.date_range("1980-01-01", "2014-12-31", freq="D")
    return [dataset_dir / f"sample_{date.strftime('%Y%m%d')}.npz" for date in dates]


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def run_step(
    *,
    name: str,
    command: list[str],
    expected_outputs: list[Path],
    cleanup_targets: list[Path],
    if_exists: str,
    dry_run: bool,
) -> None:
    expected_outputs = [Path(path) for path in expected_outputs]
    cleanup_targets = [Path(path) for path in cleanup_targets]
    complete = bool(expected_outputs) and all(path.exists() for path in expected_outputs)

    if complete and if_exists == "skip":
        print(f"[skip] {name}: expected outputs already exist")
        return

    if complete and if_exists == "overwrite":
        print(f"[overwrite] {name}: removing existing outputs")
        for target in cleanup_targets:
            if target.exists():
                remove_path(target)

    print(f"[run] {name}: {' '.join(command)}")
    if dry_run:
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    if bool(args.phase1_start_date) != bool(args.phase1_end_date):
        raise ValueError("phase1 date window requires both --phase1-start-date and --phase1-end-date")
    steps = resolve_steps(args.steps)
    exp = args.exp
    exp_cfg = CONFIG[exp]
    ssp = args.ssp or exp_cfg.get("ssp", "ssp585")
    dataset_dir = Path(exp_cfg["dataset"])
    phase1_outputs = list_phase1_outputs(dataset_dir, args.phase1_start_date, args.phase1_end_date)

    stats_outputs = [
        dataset_dir / "statistics.json",
        dataset_dir / "hist_y_train.png",
        dataset_dir / "hist_y_val.png",
        dataset_dir / "hist_y_test.png",
    ]
    prep_phase1_outputs = [
        Path(exp_cfg["target_file"]),
        Path(exp_cfg["orog_file"]),
    ]
    bc_outputs = [
        DATASET_BC_DIR / f"bc_train_hist_{args.simu}.npz",
        DATASET_BC_DIR / f"bc_test_hist_{args.simu}.npz",
        DATASET_BC_DIR / f"bc_test_future_{args.simu}.npz",
    ]
    raw_dataset_dir = DATASET_BC_DIR / f"dataset_{exp}_test_gcm"
    bc_dataset_dir = DATASET_BC_DIR / f"dataset_{exp}_test_{args.simu}_bc"
    bc_netcdf_dir = GCM_BC_DIR if args.simu == "gcm" else RCM_BC_DIR
    bc_hist_suffix = "gr" if args.simu == "gcm" else "gr_150km"
    bc_apply_outputs = [
        bc_netcdf_dir / f"{args.var}_day_{'CNRM-CM6-1' if args.simu == 'gcm' else 'ALADIN'}_historical_r1i1p1f2_{bc_hist_suffix}_19800101-19991231_bc.nc",
        bc_netcdf_dir / f"{args.var}_day_{'CNRM-CM6-1' if args.simu == 'gcm' else 'ALADIN'}_historical_r1i1p1f2_{bc_hist_suffix}_20000101-20141231_bc.nc",
        bc_netcdf_dir / f"{args.var}_day_{'CNRM-CM6-1' if args.simu == 'gcm' else 'ALADIN'}_{ssp}_r1i1p1f2_{bc_hist_suffix}_20150101-21001231_bc.nc",
        bc_dataset_dir / "sample_19800101.npz",
    ]
    pp_outputs = [bc_dataset_dir / "sample_19800101.npz"]
    raw_outputs = [
        raw_dataset_dir / "sample_20000101.npz",
        raw_dataset_dir / "sample_20150101.npz",
    ]
    prediction_test_name = f"{args.test_name}_{args.simu_test}" if args.test_name and args.simu_test else args.test_name
    prediction_period = "historical" if pd.Timestamp(args.predict_end_date) <= pd.Timestamp("2014-12-31") else ssp
    prediction_outputs = []
    if prediction_test_name:
        prediction_outputs = [
            PREDICTION_DIR / f"{args.var}_day_CNRM-CM6-1_{prediction_period}_r1i1p1f2_gr_{args.predict_start_date}_{args.predict_end_date}_{exp}_{prediction_test_name}.nc"
        ]
    value_outputs = []
    if args.test_name:
        value_outputs = [METRICS_DIR / exp / f"value_metrics_{exp}_{args.test_name}.csv"]
    metrics_test_name = f"{args.test_name}_{args.simu_test}" if args.test_name and args.simu_test else args.test_name
    metrics_day_outputs = []
    metrics_month_outputs = []
    plot_day_outputs = []
    plot_month_outputs = []
    train_outputs = []
    if metrics_test_name:
        metrics_day_outputs = [
            METRICS_DIR / exp / "mean_metrics" / f"metrics_test_daily_{exp}_{metrics_test_name}.npz",
            METRICS_DIR / exp / "mean_metrics" / f"metrics_test_mean_daily_{exp}_{metrics_test_name}.csv",
        ]
        metrics_month_outputs = [
            METRICS_DIR / exp / "mean_metrics" / f"metrics_test_monthly_{exp}_{metrics_test_name}.npz",
            METRICS_DIR / exp / "mean_metrics" / f"metrics_test_mean_monthly_{exp}_{metrics_test_name}.csv",
        ]
        plot_day_outputs = [
            GRAPHS_DIR / "metrics" / exp / metrics_test_name / f"daily_spatial_rmse_distribution_{metrics_test_name}.png",
            GRAPHS_DIR / "metrics" / exp / metrics_test_name / f"daily_spatial_bias_distribution_{metrics_test_name}.png",
            GRAPHS_DIR / "metrics" / exp / metrics_test_name / f"daily_rmse_seasonal_{metrics_test_name}.png",
        ]
        plot_month_outputs = [
            GRAPHS_DIR / "metrics" / exp / metrics_test_name / f"monthly_spatial_rmse_distribution_{metrics_test_name}.png",
            GRAPHS_DIR / "metrics" / exp / metrics_test_name / f"monthly_spatial_bias_distribution_{metrics_test_name}.png",
            GRAPHS_DIR / "metrics" / exp / metrics_test_name / f"monthly_rmse_seasonal_{metrics_test_name}.png",
        ]
    if args.test_name:
        train_outputs = [
            RUNS_DIR / exp / args.test_name / "lightning_logs" / "version_best" / "metrics_test_set.csv",
        ]

    step_table = {
        "prep_phase1": {
            "command": [
                args.python_bin,
                "bin/preprocessing/prepare_exp5_france_targets.py",
                "--exp",
                exp,
            ],
            "expected": prep_phase1_outputs,
            "cleanup": prep_phase1_outputs,
        },
        "phase1": {
            "command": [
                args.python_bin,
                "bin/preprocessing/build_dataset.py",
                "--exp",
                exp,
            ],
            "expected": phase1_outputs,
            "cleanup": phase1_outputs,
        },
        "stats": {
            "command": [args.python_bin, "bin/preprocessing/compute_statistics.py", "--exp", exp],
            "expected": stats_outputs,
            "cleanup": stats_outputs,
        },
        "bc_dataset": {
            "command": [
                args.python_bin,
                "bin/preprocessing/build_dataset_bc.py",
                "--exp",
                exp,
                "--simu",
                args.simu,
                "--ssp",
                ssp,
                "--var",
                args.var,
            ],
            "expected": bc_outputs,
            "cleanup": bc_outputs,
        },
        "bc_apply": {
            "command": [
                args.python_bin,
                "bin/preprocessing/bias_correction_ibicus.py",
                "--exp",
                exp,
                "--ssp",
                ssp,
                "--simu",
                args.simu,
                "--var",
                args.var,
            ],
            "expected": bc_apply_outputs,
            "cleanup": bc_apply_outputs + [bc_dataset_dir, GRAPHS_DIR / "biascorrection"],
        },
        "raw_dataset": {
            "command": [
                args.python_bin,
                "bin/preprocessing/build_dataset_pp.py",
                "--exp",
                exp,
                "--simu",
                "gcm",
                "--var",
                args.var,
            ],
            "expected": raw_outputs,
            "cleanup": [raw_dataset_dir],
        },
        "train": {
            "command": [
                args.python_bin,
                "bin/training/train.py",
                "--exp",
                exp,
                "--test-name",
                args.test_name or "",
                "--model",
                args.train_model,
                "--max-epoch",
                str(args.train_max_epoch),
                "--batch-size",
                str(args.train_batch_size),
                "--learning-rate",
                str(args.train_learning_rate),
            ]
            + (["--loss", args.train_loss] if args.train_loss else [])
            + (["--output-norm"] if args.train_output_norm else []),
            "expected": train_outputs,
            "cleanup": [RUNS_DIR / exp / (args.test_name or "")],
        },
        "pp_dataset": {
            "command": [
                args.python_bin,
                "bin/preprocessing/build_dataset_pp.py",
                "--exp",
                exp,
                "--simu",
                "gcm",
                "--var",
                args.var,
                "--corrected",
            ],
            "expected": pp_outputs,
            "cleanup": [bc_dataset_dir],
        },
        "predict_loop": {
            "command": [
                args.python_bin,
                "bin/training/predict_loop.py",
                "--exp",
                exp,
                "--test-name",
                args.test_name or "",
                "--simu-test",
                args.simu_test,
                "--startdate",
                args.predict_start_date,
                "--enddate",
                args.predict_end_date,
            ] + (["--checkpoint-bundle", args.checkpoint_bundle] if args.checkpoint_bundle else []),
            "expected": prediction_outputs,
            "cleanup": prediction_outputs,
        },
        "value_metrics": {
            "command": [
                args.python_bin,
                "bin/evaluation/compute_value_metrics.py",
                "--exp",
                exp,
                "--test-name",
                args.test_name or "",
                "--simu-test",
                args.simu_test,
                "--simu",
                args.simu,
            ],
            "expected": value_outputs,
            "cleanup": value_outputs,
        },
        "metrics_day": {
            "command": [
                args.python_bin,
                "bin/evaluation/compute_test_metrics_day.py",
                "--startdate",
                args.predict_start_date,
                "--enddate",
                args.predict_end_date,
                "--exp",
                exp,
                "--test-name",
                args.test_name or "",
                "--simu-test",
                args.simu_test,
            ] + (["--checkpoint-bundle", args.checkpoint_bundle] if args.checkpoint_bundle else []),
            "expected": metrics_day_outputs,
            "cleanup": metrics_day_outputs,
        },
        "metrics_month": {
            "command": [
                args.python_bin,
                "bin/evaluation/compute_test_metrics_month.py",
                "--startdate",
                args.predict_start_date,
                "--enddate",
                args.predict_end_date,
                "--exp",
                exp,
                "--test-name",
                args.test_name or "",
                "--simu-test",
                args.simu_test,
            ] + (["--checkpoint-bundle", args.checkpoint_bundle] if args.checkpoint_bundle else []),
            "expected": metrics_month_outputs,
            "cleanup": metrics_month_outputs,
        },
        "plot_metrics_day": {
            "command": [
                args.python_bin,
                "bin/evaluation/plot_test_metrics.py",
                "--exp",
                exp,
                "--test-name",
                metrics_test_name or "",
                "--scale",
                "daily",
            ],
            "expected": plot_day_outputs,
            "cleanup": plot_day_outputs,
        },
        "plot_metrics_month": {
            "command": [
                args.python_bin,
                "bin/evaluation/plot_test_metrics.py",
                "--exp",
                exp,
                "--test-name",
                metrics_test_name or "",
                "--scale",
                "monthly",
            ],
            "expected": plot_month_outputs,
            "cleanup": plot_month_outputs,
        },
    }

    if args.phase1_start_date:
        step_table["phase1"]["command"].extend(["--start_date", args.phase1_start_date])
    if args.phase1_end_date:
        step_table["phase1"]["command"].extend(["--end_date", args.phase1_end_date])
    if "pp_dataset" in steps and args.simu != "gcm":
        raise ValueError("pp_dataset currently supports only --simu gcm")
    if "raw_dataset" in steps and args.simu != "gcm":
        raise ValueError("raw_dataset currently supports only --simu gcm")
    if "train" in steps and not args.test_name:
        raise ValueError("--test-name is required for the train step")
    if any(step in steps for step in ["predict_loop", "metrics_day", "metrics_month", "value_metrics", "plot_metrics_day", "plot_metrics_month"]) and not args.test_name:
        raise ValueError("--test-name is required for predict_loop, metrics_day, metrics_month, value_metrics, plot_metrics_day, and plot_metrics_month steps")

    for step in steps:
        run_step(
            name=step,
            command=step_table[step]["command"],
            expected_outputs=step_table[step]["expected"],
            cleanup_targets=step_table[step]["cleanup"],
            if_exists=args.if_exists,
            dry_run=args.dry_run,
        )

    print("[done] exp5 workflow orchestration finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
