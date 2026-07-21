#!/usr/bin/env python3
"""
Run the observation-target workflow with simple step orchestration.

This entrypoint keeps the clean branch usable for day-to-day work:
- central observation-target workflow command
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

from iriscc.settings import (
    CONFIG,
    GRAPHS_DIR,
    METRICS_DIR,
    RUNS_DIR,
    get_bc_bundle_path,
    get_bc_train_hist_dates,
    get_bc_test_future_dates,
    get_bc_test_hist_dates,
    get_bias_corrected_sample_dir,
    get_dataset_variant_dir,
    get_experiment_prediction_frequency,
    get_experiment_training_frequency,
    get_frequency_pandas_rule,
    get_metrics_test_name,
    get_bias_corrected_netcdf_path,
    get_phase1_dates,
    get_prediction_output_path,
    get_source_default_frequency,
    format_sample_time_token,
)
from iriscc.provenance import build_prov_bundle, inventory_paths, print_resolved_context, utc_now_iso, write_provjson


DEFAULT_STEPS = ["phase1", "stats", "bc_dataset", "bc_apply"]
OPTIONAL_STEPS = [
    "prep_phase1",
    "train",
    "raw_dataset",
    "pp_dataset",
    "derive_products",
    "predict_loop",
    "metrics_day",
    "metrics_month",
    "value_metrics",
    "plot_metrics_day",
    "plot_metrics_month",
    "compare_suite",
]
ALL_STEPS = DEFAULT_STEPS + OPTIONAL_STEPS


def yyyymmdd(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def default_phase1_window(exp: str) -> tuple[str, str]:
    dates = get_phase1_dates(exp)
    return yyyymmdd(dates[0]), yyyymmdd(dates[-1])


def default_prediction_window(exp: str) -> tuple[str, str]:
    hist_dates = get_bc_test_hist_dates(exp)
    future_dates = get_bc_test_future_dates(exp)
    return yyyymmdd(hist_dates[0]), yyyymmdd(future_dates[-1])


def default_metrics_window(exp: str) -> tuple[str, str]:
    hist_dates = get_bc_test_hist_dates(exp)
    return yyyymmdd(hist_dates[0]), yyyymmdd(hist_dates[-1])


def default_value_window(exp: str) -> tuple[str, str]:
    hist_dates = get_bc_test_hist_dates(exp)
    return yyyymmdd(hist_dates[0]), yyyymmdd(hist_dates[-1])


def first_hist_train_day(exp: str) -> str:
    dates = get_bc_train_hist_dates(exp)
    return yyyymmdd(dates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the observation-target workflow end to end.")
    parser.add_argument("--exp", default="exp5", help="Experiment name. Supports exp5, expc, and related obs-target workflows.")
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
    parser.add_argument(
        "--simu",
        default="gcm",
        help="Simulation alias or model source key for phase2 steps, e.g. gcm, rcm, cordex, gcm_cnrm_cm6_1.",
    )
    parser.add_argument("--var", default="tas", help="Variable for phase2 steps.")
    parser.add_argument("--paired-vars", default=None, help="Optional comma-separated pair of variables for paired multivariate BC methods.")
    parser.add_argument("--derive-wind-products", action="store_true", help="Derive scalar wind products from paired corrected components.")
    parser.add_argument("--u-var", default="uas", help="Zonal wind-component variable name used for wind-product derivation.")
    parser.add_argument("--v-var", default="vas", help="Meridional wind-component variable name used for wind-product derivation.")
    parser.add_argument("--speed-var", default="sfcWind", help="Derived wind-speed variable name.")
    parser.add_argument("--direction-var", default="windFromDirection", help="Derived wind-direction variable name.")
    parser.add_argument("--ssp", default=None, help="Override SSP scenario. Defaults to experiment config.")
    parser.add_argument(
        "--bc-method",
        default=None,
        help="Bias-correction method override. Supported production methods: ibicus_cdft, sbck_cdft, sbck_mbcn.",
    )
    parser.add_argument("--test-name", default=None, help="Model run name for inference/evaluation steps.")
    parser.add_argument("--checkpoint-bundle", default=None, help="Optional portable checkpoint bundle directory for inference/evaluation.")
    parser.add_argument("--train-max-epoch", type=int, default=30, help="Max epochs for the optional train step.")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Batch size for the optional train step.")
    parser.add_argument("--train-learning-rate", type=float, default=8e-4, help="Learning rate for the optional train step.")
    parser.add_argument("--train-model", default="unet", help="Model family for the optional train step.")
    parser.add_argument("--train-loss", default=None, help="Optional loss override for the train step.")
    parser.add_argument("--train-output-norm", action="store_true", help="Enable output normalization in the train step.")
    parser.add_argument("--sample-start-date", default=None, help="Optional YYYYMMDD start date for raw/BC sample packaging.")
    parser.add_argument("--sample-end-date", default=None, help="Optional YYYYMMDD end date for raw/BC sample packaging.")
    parser.add_argument("--predict-start-date", default=None, help="Prediction start date. Defaults to the settings.py historical-test start.")
    parser.add_argument("--predict-end-date", default=None, help="Prediction end date. Defaults to the settings.py future-test end.")
    parser.add_argument("--metrics-start-date", default=None, help="Metrics evaluation start date. Defaults to the settings.py historical-test start.")
    parser.add_argument("--metrics-end-date", default=None, help="Metrics evaluation end date. Defaults to the settings.py historical-test end.")
    parser.add_argument("--value-start-date", default=None, help="VALUE validation start date. Defaults to the settings.py historical-test start.")
    parser.add_argument("--value-end-date", default=None, help="VALUE validation end date. Defaults to the settings.py historical-test end.")
    parser.add_argument(
        "--simu-test",
        default="gcm_bc",
        help="Inference sample variant, e.g. gcm, gcm_bc, cordex, or cordex_bc.",
    )
    parser.add_argument(
        "--compare-models",
        default=None,
        help="Comma-separated ML test names to include in default raw/BC/ML comparison diagnostics. Defaults to --test-name.",
    )
    parser.add_argument(
        "--skip-default-comparisons",
        action="store_true",
        help="Disable the default raw/BC/ML comparison suite when evaluation steps are requested.",
    )
    parser.add_argument(
        "--compare-stride-days",
        type=int,
        default=7,
        help="Sampling stride in days for the comparison distribution plot.",
    )
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


def resolve_compare_models(test_name: str | None, raw_value: str | None) -> list[str]:
    if raw_value:
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if test_name:
        return [test_name]
    return []


def maybe_add_default_comparison_step(steps: list[str], args: argparse.Namespace) -> list[str]:
    if args.skip_default_comparisons:
        return steps
    evaluation_steps = {
        "metrics_day",
        "metrics_month",
        "value_metrics",
        "plot_metrics_day",
        "plot_metrics_month",
    }
    if any(step in steps for step in evaluation_steps) and "compare_suite" not in steps:
        return steps + ["compare_suite"]
    return steps


def list_phase1_outputs(exp: str, dataset_dir: Path, start: str | None, end: str | None) -> list[Path]:
    training_frequency = get_experiment_training_frequency(exp)
    if training_frequency != "daily":
        raise ValueError(
            f"Phase-1 sample packaging currently expects daily outputs, got '{training_frequency}' for experiment '{exp}'."
        )
    if start and end:
        dates = pd.date_range(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            freq=get_frequency_pandas_rule(training_frequency),
        )
    else:
        dates = get_phase1_dates(exp)
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
    start_time = utc_now_iso()
    args = parse_args()
    if bool(args.phase1_start_date) != bool(args.phase1_end_date):
        raise ValueError("phase1 date window requires both --phase1-start-date and --phase1-end-date")
    steps = resolve_steps(args.steps)
    steps = maybe_add_default_comparison_step(steps, args)
    exp = args.exp
    exp_cfg = CONFIG[exp]
    ssp = args.ssp or exp_cfg.get("ssp", "ssp585")
    bc_method = args.bc_method or exp_cfg.get("bias_correction_method", "ibicus_cdft")
    paired_vars = [item.strip() for item in (args.paired_vars or "").split(",") if item.strip()]
    bc_output_tag = "sbck_mbcn" if bc_method == "sbck_mbcn" else None
    workflow_simu_test = args.simu_test
    default_corrected_variant = f"{args.simu}_bc"
    if bc_output_tag and workflow_simu_test == default_corrected_variant:
        workflow_simu_test = f"{default_corrected_variant}_{bc_output_tag}"
    bc_apply_script = {
        "ibicus_cdft": "bin/preprocessing/bias_correction_ibicus.py",
        "sbck_cdft": "bin/preprocessing/bias_correction_sbck.py",
        "sbck_mbcn": "bin/preprocessing/bias_correction_sbck_mbcn.py",
    }.get(bc_method)
    phase1_start_date, phase1_end_date = (
        (args.phase1_start_date, args.phase1_end_date)
        if args.phase1_start_date and args.phase1_end_date
        else default_phase1_window(exp)
    )
    predict_start_date = args.predict_start_date or default_prediction_window(exp)[0]
    predict_end_date = args.predict_end_date or default_prediction_window(exp)[1]
    metrics_start_date = args.metrics_start_date or default_metrics_window(exp)[0]
    metrics_end_date = args.metrics_end_date or default_metrics_window(exp)[1]
    value_start_date = args.value_start_date or default_value_window(exp)[0]
    value_end_date = args.value_end_date or default_value_window(exp)[1]
    dataset_dir = Path(exp_cfg["dataset"])
    resolved_settings = {
        "exp": exp,
        "steps": steps,
        "compare_models": resolve_compare_models(args.test_name, args.compare_models),
        "dataset_dir": dataset_dir,
        "ssp": ssp,
        "bc_method": bc_method,
        "paired_vars": paired_vars,
        "simu": args.simu,
        "simu_test": workflow_simu_test,
        "training_frequency": get_experiment_training_frequency(exp),
        "prediction_frequency": get_experiment_prediction_frequency(exp),
        "target_default_frequency": get_source_default_frequency(exp_cfg.get("target_source", exp_cfg["target"])),
    }
    path_inventory = inventory_paths(
        {
            "dataset_dir": dataset_dir,
            "runs_dir": RUNS_DIR / exp,
            "metrics_dir": METRICS_DIR / exp,
            "graphs_dir": GRAPHS_DIR / exp,
        }
    )
    print_resolved_context(
        script_name="run_obs_workflow.py",
        parameters=vars(args),
        settings={**resolved_settings, "path_inventory": path_inventory},
        inputs={"dataset_dir": dataset_dir},
        outputs={"runs_dir": RUNS_DIR / exp, "metrics_dir": METRICS_DIR / exp, "graphs_dir": GRAPHS_DIR / exp},
    )
    phase1_outputs = list_phase1_outputs(exp, dataset_dir, phase1_start_date, phase1_end_date)

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
    bc_bundle_variables = paired_vars if bc_method == "sbck_mbcn" else None
    bc_outputs = [
        get_bc_bundle_path(exp, args.simu, "train_hist", variables=bc_bundle_variables),
        get_bc_bundle_path(exp, args.simu, "test_hist", variables=bc_bundle_variables),
        get_bc_bundle_path(exp, args.simu, "test_future", variables=bc_bundle_variables),
    ]
    raw_dataset_dir = get_dataset_variant_dir(exp, args.simu)
    bc_dataset_dir = get_bias_corrected_sample_dir(exp, args.simu, bc_tag=bc_output_tag)
    if bc_method == "sbck_mbcn":
        bc_apply_outputs = [
            get_bias_corrected_netcdf_path(exp, args.simu, var, "train_hist", ssp=ssp, bc_tag=bc_output_tag)
            for var in paired_vars
        ] + [
            get_bias_corrected_netcdf_path(exp, args.simu, var, "test_hist", ssp=ssp, bc_tag=bc_output_tag)
            for var in paired_vars
        ] + [
            get_bias_corrected_netcdf_path(exp, args.simu, var, "test_future", ssp=ssp, bc_tag=bc_output_tag)
            for var in paired_vars
        ]
    else:
        bc_apply_outputs = [
            get_bias_corrected_netcdf_path(exp, args.simu, args.var, "train_hist", ssp=ssp),
            get_bias_corrected_netcdf_path(exp, args.simu, args.var, "test_hist", ssp=ssp),
            get_bias_corrected_netcdf_path(exp, args.simu, args.var, "test_future", ssp=ssp),
            bc_dataset_dir / f"sample_{first_hist_train_day(exp)}.npz",
        ]
    pp_outputs = [
        bc_dataset_dir
        / f"sample_{format_sample_time_token(get_bc_test_hist_dates(exp)[0], get_experiment_prediction_frequency(exp))}.npz"
    ]
    raw_outputs = [
        raw_dataset_dir
        / f"sample_{format_sample_time_token(get_bc_test_hist_dates(exp)[0], get_experiment_prediction_frequency(exp))}.npz",
        raw_dataset_dir
        / f"sample_{format_sample_time_token(get_bc_test_future_dates(exp)[0], get_experiment_prediction_frequency(exp))}.npz",
    ]
    prediction_test_name = get_metrics_test_name(args.test_name, workflow_simu_test) if args.test_name else None
    prediction_outputs = []
    if prediction_test_name:
        prediction_outputs = [
            get_prediction_output_path(
                exp,
                workflow_simu_test,
                args.var,
                predict_start_date,
                predict_end_date,
                prediction_test_name,
                ssp=ssp,
            )
        ]
    value_outputs = []
    if args.test_name:
        value_outputs = [METRICS_DIR / exp / f"value_metrics_{exp}_{args.test_name}.csv"]
    metrics_test_name = get_metrics_test_name(args.test_name, workflow_simu_test) if args.test_name else None
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
            ] + (["--paired-vars", ",".join(paired_vars)] if bc_method == "sbck_mbcn" else []),
            "expected": bc_outputs,
            "cleanup": bc_outputs,
        },
        "bc_apply": {
            "command": [
                args.python_bin,
                bc_apply_script or "",
                "--exp",
                exp,
                "--ssp",
                ssp,
                "--simu",
                args.simu,
                "--var",
                args.var,
            ] + (["--paired-vars", ",".join(paired_vars)] if bc_method == "sbck_mbcn" else []),
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
                args.simu,
                "--var",
                args.var,
                "--ssp",
                ssp,
            ]
            + (["--startdate", args.sample_start_date] if args.sample_start_date else [])
            + (["--enddate", args.sample_end_date] if args.sample_end_date else []),
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
                args.simu,
                "--var",
                args.var,
                "--ssp",
                ssp,
                "--corrected",
            ]
            + (["--bc-tag", bc_output_tag] if bc_output_tag else [])
            + (["--startdate", args.sample_start_date] if args.sample_start_date else [])
            + (["--enddate", args.sample_end_date] if args.sample_end_date else []),
            "expected": pp_outputs,
            "cleanup": [bc_dataset_dir],
        },
        "derive_products": {
            "command": [
                args.python_bin,
                "bin/postprocessing/run_derive_wind_products_from_bc.py",
                "--exp",
                exp,
                "--simu",
                args.simu,
                "--ssp",
                ssp,
                "--u-var",
                args.u_var,
                "--v-var",
                args.v_var,
                "--speed-var",
                args.speed_var,
                "--bc-tag",
                bc_output_tag or "sbck_mbcn",
                "--python-bin",
                args.python_bin,
            ]
            + (["--direction", "--direction-var", args.direction_var] if args.derive_wind_products else []),
            "expected": [
                get_bias_corrected_netcdf_path(exp, args.simu, args.speed_var, "train_hist", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
                get_bias_corrected_netcdf_path(exp, args.simu, args.speed_var, "test_hist", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
                get_bias_corrected_netcdf_path(exp, args.simu, args.speed_var, "test_future", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
            ]
            + (
                [
                    get_bias_corrected_netcdf_path(exp, args.simu, args.direction_var, "train_hist", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
                    get_bias_corrected_netcdf_path(exp, args.simu, args.direction_var, "test_hist", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
                    get_bias_corrected_netcdf_path(exp, args.simu, args.direction_var, "test_future", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
                ]
                if args.derive_wind_products
                else []
            ),
            "cleanup": [
                get_bias_corrected_netcdf_path(exp, args.simu, args.speed_var, "train_hist", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
                get_bias_corrected_netcdf_path(exp, args.simu, args.speed_var, "test_hist", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
                get_bias_corrected_netcdf_path(exp, args.simu, args.speed_var, "test_future", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
            ]
            + (
                [
                    get_bias_corrected_netcdf_path(exp, args.simu, args.direction_var, "train_hist", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
                    get_bias_corrected_netcdf_path(exp, args.simu, args.direction_var, "test_hist", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
                    get_bias_corrected_netcdf_path(exp, args.simu, args.direction_var, "test_future", ssp=ssp, bc_tag=bc_output_tag or "sbck_mbcn"),
                ]
                if args.derive_wind_products
                else []
            ),
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
                workflow_simu_test,
                "--startdate",
                predict_start_date,
                "--enddate",
                predict_end_date,
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
                workflow_simu_test,
                "--simu",
                args.simu,
                "--startdate",
                value_start_date,
                "--enddate",
                value_end_date,
            ],
            "expected": value_outputs,
            "cleanup": value_outputs,
        },
        "metrics_day": {
            "command": [
                args.python_bin,
                "bin/evaluation/compute_test_metrics_day.py",
                "--startdate",
                metrics_start_date,
                "--enddate",
                metrics_end_date,
                "--exp",
                exp,
                "--test-name",
                args.test_name or "",
                "--simu-test",
                workflow_simu_test,
            ] + (["--checkpoint-bundle", args.checkpoint_bundle] if args.checkpoint_bundle else []),
            "expected": metrics_day_outputs,
            "cleanup": metrics_day_outputs,
        },
        "metrics_month": {
            "command": [
                args.python_bin,
                "bin/evaluation/compute_test_metrics_month.py",
                "--startdate",
                metrics_start_date,
                "--enddate",
                metrics_end_date,
                "--exp",
                exp,
                "--test-name",
                args.test_name or "",
                "--simu-test",
                workflow_simu_test,
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
        "compare_suite": {
            "command": [
                args.python_bin,
                "bin/evaluation/run_obs_comparison_suite.py",
                "--exp",
                exp,
                "--simu",
                args.simu,
                "--simu-test",
                workflow_simu_test,
                "--var",
                args.var,
                "--startdate",
                metrics_start_date,
                "--enddate",
                metrics_end_date,
                "--value-startdate",
                value_start_date,
                "--value-enddate",
                value_end_date,
                "--ml-models",
                ",".join(resolve_compare_models(args.test_name, args.compare_models)),
                "--python-bin",
                args.python_bin,
                "--stride-days",
                str(args.compare_stride_days),
            ],
            "expected": [],
            "cleanup": [],
        },
    }

    step_table["phase1"]["command"].extend(["--start_date", phase1_start_date, "--end_date", phase1_end_date])
    if bc_apply_script is None and "bc_apply" in steps:
        raise NotImplementedError(
            f"Bias-correction method '{bc_method}' is not wired into the production workflow yet. "
            "Supported production methods are 'ibicus_cdft', 'sbck_cdft', and 'sbck_mbcn'."
        )
    if bc_method == "sbck_mbcn":
        if len(paired_vars) != 2:
            raise ValueError("--paired-vars is required with exactly two variables for --bc-method sbck_mbcn.")
        if args.var == args.direction_var and any(
            step in steps
            for step in {"metrics_day", "metrics_month", "value_metrics", "plot_metrics_day", "plot_metrics_month", "compare_suite"}
        ):
            raise NotImplementedError("windFromDirection requires dedicated circular diagnostics and is not supported by the default scalar evaluation steps.")
        scalar_downstream = {"pp_dataset", "train", "predict_loop", "metrics_day", "metrics_month", "value_metrics", "plot_metrics_day", "plot_metrics_month", "compare_suite"}
        requested_scalar_downstream = [step for step in steps if step in scalar_downstream]
        if requested_scalar_downstream and not ("derive_products" in steps and args.var == args.speed_var):
            raise NotImplementedError(
                "sbck_mbcn scalar downstream steps currently require deriving a scalar product first. "
                f"Requested steps: {', '.join(requested_scalar_downstream)}. "
                f"Use --derive-wind-products with --var {args.speed_var} and include derive_products in --steps."
            )
    if "train" in steps and not args.test_name:
        raise ValueError("--test-name is required for the train step")
    evaluation_steps = [
        "predict_loop",
        "metrics_day",
        "metrics_month",
        "value_metrics",
        "plot_metrics_day",
        "plot_metrics_month",
    ]
    if any(step in steps for step in evaluation_steps) and not args.test_name:
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

    print("[done] observation-target workflow orchestration finished")
    prov_path = write_provjson(
        METRICS_DIR / exp / f"workflow_{args.test_name or 'no_test_name'}.prov.json",
        build_prov_bundle(
            script_name="run_obs_workflow.py",
            activity_type="workflow",
            start_time=start_time,
            end_time=utc_now_iso(),
            parameters=vars(args),
            settings={**resolved_settings, "path_inventory": path_inventory},
            inputs={"dataset_dir": dataset_dir},
            outputs={"runs_dir": RUNS_DIR / exp, "metrics_dir": METRICS_DIR / exp, "graphs_dir": GRAPHS_DIR / exp},
            cwd=PROJECT_ROOT,
        ),
    )
    print(f"provenance_provjson={prov_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
