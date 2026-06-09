#!/usr/bin/env python3
"""
Standalone RCM perfect-model workflow.

This path trains on a historical degraded-RCM-to-native-RCM dataset built from
a configurable coarse bridge grid and evaluates ML output against the
fine-resolution RCM pseudo-truth.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iriscc.settings import (
    CONFIG,
    DATASET_BC_DIR,
    DATES_BC_TEST_HIST,
    DATES_BC_TRAIN_HIST,
    GRAPHS_DIR,
    METRICS_DIR,
    RUNS_DIR,
    get_prediction_output_path,
    get_value_metrics_output_path,
)


DEFAULT_STEPS = [
    "build_train_dataset",
    "build_eval_dataset",
    "validate_train_dataset",
    "validate_eval_dataset",
    "stats",
    "train",
    "predict",
    "compare_predictions",
    "aggregate_comparison",
    "plot_score_comparison",
    "plot_distribution",
    "metrics_ml_day",
    "metrics_ml_month",
    "value_metrics",
    "metrics_rcm_day",
    "metrics_rcm_month",
]


def yyyymmdd(value) -> str:
    return value.strftime("%Y%m%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone RCM perfect-model workflow.")
    parser.add_argument("--exp", default="perfect_model_rcm")
    parser.add_argument("--var", default="tas")
    parser.add_argument("--ssp", default="ssp585")
    parser.add_argument("--test-name", default="unet_perfect_model_rcm")
    parser.add_argument("--simu", default="rcm", help="Simulation family or source key for the perfect-model path.")
    parser.add_argument("--steps", default="all", help=f"Comma-separated steps or 'all' for {','.join(DEFAULT_STEPS)}.")
    parser.add_argument("--if-exists", choices=["skip", "overwrite"], default="skip")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--train-max-epoch", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--predict-batch-size", type=int, default=None)
    parser.add_argument("--train-learning-rate", type=float, default=8e-4)
    parser.add_argument("--train-model", default="unet")
    parser.add_argument("--train-loss", default=None)
    parser.add_argument("--train-output-norm", action="store_true")
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--train-n-steps", type=int, default=200)
    parser.add_argument("--sample-dir", default=None, help="Optional perfect-model sample directory override.")
    parser.add_argument("--perfect-model-target-source", default=None, help="Optional native target source override for perfect-model sample generation.")
    parser.add_argument("--validation-startdate", default=None, help="Optional start date for training-sample validation.")
    parser.add_argument("--validation-enddate", default=None, help="Optional end date for training-sample validation.")
    parser.add_argument("--validation-historical-enddate", default=None, help="Optional historical cutoff for validation windows.")
    parser.add_argument("--validation-unit", default="", help="Optional unit label for validation reports and plots.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_steps(raw_steps: str) -> list[str]:
    if raw_steps == "all":
        return list(DEFAULT_STEPS)
    steps = [step.strip() for step in raw_steps.split(",") if step.strip()]
    unknown = [step for step in steps if step not in DEFAULT_STEPS]
    if unknown:
        raise ValueError(f"Unknown steps: {', '.join(unknown)}")
    return steps


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
    complete = bool(expected_outputs) and all(Path(path).exists() for path in expected_outputs)
    if complete and if_exists == "skip":
        print(f"[skip] {name}: expected outputs already exist")
        return
    if complete and if_exists == "overwrite":
        print(f"[overwrite] {name}: removing existing outputs")
        for target in cleanup_targets:
            if Path(target).exists():
                remove_path(Path(target))
    print(f"[run] {name}: {' '.join(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    steps = resolve_steps(args.steps)
    train_start = yyyymmdd(DATES_BC_TRAIN_HIST[0])
    hist_start = yyyymmdd(DATES_BC_TEST_HIST[0])
    hist_end = yyyymmdd(DATES_BC_TEST_HIST[-1])
    validation_start = args.validation_startdate or train_start
    validation_end = args.validation_enddate or hist_end
    validation_historical_end = args.validation_historical_enddate or hist_end
    perfect_model_target_source = args.perfect_model_target_source or CONFIG[args.exp].get("perfect_model_target_source", "rcm_aladin")
    train_output_norm = args.train_output_norm or args.train_model == "cddpm"

    perfect_dataset_dir = Path(args.sample_dir) if args.sample_dir else Path(CONFIG[args.exp]["dataset"])
    eval_dataset_dir = DATASET_BC_DIR / f"dataset_{args.exp}_test_{args.simu}"
    runs_dir = RUNS_DIR / args.exp / args.test_name
    metrics_test_name = f"{args.test_name}_{args.simu}"
    prediction_path = get_prediction_output_path(
        args.exp,
        args.simu,
        args.var,
        hist_start,
        hist_end,
        metrics_test_name,
        ssp=args.ssp,
    )
    validation_output_dir = METRICS_DIR / args.exp / "validation"
    comparison_output_dir = METRICS_DIR / args.exp / "comparison_tables"
    validation_train_stem = f"perfect_model_samples_{args.exp}_{args.simu}_{validation_start}_{validation_end}"
    validation_eval_stem = f"perfect_model_samples_{args.exp}_{args.simu}_{hist_start}_{hist_end}"
    combined_comparison_stem = f"perfect_model_predictions_vs_truth_{args.exp}_combined_{args.simu}"
    step_table = {
        "build_train_dataset": {
            "command": [
                args.python_bin,
                "bin/preprocessing/build_dataset_pp.py",
                "--exp",
                args.exp,
                "--simu",
                args.simu,
                "--var",
                args.var,
                "--ssp",
                args.ssp,
                "--perfect-model-target-source",
                perfect_model_target_source,
                "--include-train-hist",
                "--historical-only",
                "--output_dir",
                str(perfect_dataset_dir),
            ],
            "expected": [
                perfect_dataset_dir / f"sample_{train_start}.npz",
                perfect_dataset_dir / f"sample_{hist_start}.npz",
                perfect_dataset_dir / f"sample_{hist_end}.npz",
            ],
            "cleanup": [perfect_dataset_dir],
        },
        "build_eval_dataset": {
            "command": [
                args.python_bin,
                "bin/preprocessing/build_dataset_pp.py",
                "--exp",
                args.exp,
                "--simu",
                args.simu,
                "--var",
                args.var,
                "--ssp",
                args.ssp,
                "--perfect-model-target-source",
                perfect_model_target_source,
                "--historical-only",
            ],
            "expected": [
                eval_dataset_dir / f"sample_{hist_start}.npz",
                eval_dataset_dir / f"sample_{hist_end}.npz",
            ],
            "cleanup": [eval_dataset_dir],
        },
        "validate_train_dataset": {
            "command": [
                args.python_bin,
                "bin/evaluation/validate_perfect_model_samples.py",
                "--exp",
                args.exp,
                "--simu",
                args.simu,
                "--var",
                args.var,
                "--unit",
                args.validation_unit,
                "--startdate",
                validation_start,
                "--enddate",
                validation_end,
                "--historical-enddate",
                validation_historical_end,
                "--sample-dir",
                str(perfect_dataset_dir),
                "--allow-extra-samples",
            ],
            "expected": [validation_output_dir / f"{validation_train_stem}.md"],
            "cleanup": [
                validation_output_dir / f"{validation_train_stem}.md",
                validation_output_dir / f"{validation_train_stem}_inventory.csv",
                validation_output_dir / f"{validation_train_stem}_structure.csv",
                validation_output_dir / f"{validation_train_stem}_cross_period_repeat.csv",
                validation_output_dir / f"{validation_train_stem}_period_stats.csv",
                validation_output_dir / f"{validation_train_stem}_edge_stats.csv",
            ],
        },
        "validate_eval_dataset": {
            "command": [
                args.python_bin,
                "bin/evaluation/validate_perfect_model_samples.py",
                "--exp",
                args.exp,
                "--simu",
                args.simu,
                "--var",
                args.var,
                "--unit",
                args.validation_unit,
                "--startdate",
                hist_start,
                "--enddate",
                hist_end,
                "--historical-enddate",
                hist_end,
                "--sample-dir",
                str(eval_dataset_dir),
            ],
            "expected": [validation_output_dir / f"{validation_eval_stem}.md"],
            "cleanup": [
                validation_output_dir / f"{validation_eval_stem}.md",
                validation_output_dir / f"{validation_eval_stem}_inventory.csv",
                validation_output_dir / f"{validation_eval_stem}_structure.csv",
                validation_output_dir / f"{validation_eval_stem}_cross_period_repeat.csv",
                validation_output_dir / f"{validation_eval_stem}_period_stats.csv",
                validation_output_dir / f"{validation_eval_stem}_edge_stats.csv",
            ],
        },
        "stats": {
            "command": [
                args.python_bin,
                "bin/preprocessing/compute_statistics.py",
                "--exp",
                args.exp,
                "--dataset_dir",
                str(perfect_dataset_dir),
            ],
            "expected": [perfect_dataset_dir / "statistics.json"],
            "cleanup": [
                perfect_dataset_dir / "statistics.json",
                perfect_dataset_dir / "hist_y_train.png",
                perfect_dataset_dir / "hist_y_val.png",
                perfect_dataset_dir / "hist_y_test.png",
            ],
        },
        "train": {
            "command": [
                args.python_bin,
                "bin/training/train.py",
                "--exp",
                args.exp,
                "--test-name",
                args.test_name,
                "--model",
                args.train_model,
                "--max-epoch",
                str(args.train_max_epoch),
                "--batch-size",
                str(args.train_batch_size),
                "--learning-rate",
                str(args.train_learning_rate),
                "--sample-dir",
                str(perfect_dataset_dir),
                "--skip-test",
            ]
            + (["--loss", args.train_loss] if args.train_loss else [])
            + (["--output-norm"] if train_output_norm else [])
            + (["--seed", str(args.train_seed)] if args.train_seed is not None else [])
            + (["--n-steps", str(args.train_n_steps)] if args.train_model == "cddpm" else []),
            "expected": [runs_dir / "lightning_logs" / "version_best" / "metrics.csv"],
            "cleanup": [runs_dir],
        },
        "predict": {
            "command": [
                args.python_bin,
                "bin/training/predict_loop.py",
                "--exp",
                args.exp,
                "--test-name",
                args.test_name,
                "--simu-test",
                args.simu,
                "--var",
                args.var,
                "--startdate",
                hist_start,
                "--enddate",
                hist_end,
            ]
            + (["--batch-size", str(args.predict_batch_size)] if args.predict_batch_size is not None else []),
            "expected": [prediction_path],
            "cleanup": [prediction_path],
        },
        "compare_predictions": {
            "command": [
                args.python_bin,
                "bin/evaluation/compare_perfect_model_predictions_vs_truth.py",
                "--exp",
                args.exp,
                "--test-name",
                args.test_name,
                "--simu-test",
                args.simu,
                "--var",
                args.var,
                "--unit",
                args.validation_unit,
                "--startdate",
                hist_start,
                "--enddate",
                hist_end,
                "--sample-dir",
                str(eval_dataset_dir),
                "--raw-sample-dir",
                str(eval_dataset_dir),
                "--output-dir",
                str(comparison_output_dir / "chunks"),
                "--stem-suffix",
                f"_{hist_start}_{hist_end}",
            ],
            "expected": [
                comparison_output_dir
                / "chunks"
                / f"perfect_model_predictions_vs_truth_{args.exp}_{metrics_test_name}_{hist_start}_{hist_end}.csv"
            ],
            "cleanup": [
                comparison_output_dir
                / "chunks"
                / f"perfect_model_predictions_vs_truth_{args.exp}_{metrics_test_name}_{hist_start}_{hist_end}.csv",
                comparison_output_dir
                / "chunks"
                / f"perfect_model_predictions_vs_truth_{args.exp}_{metrics_test_name}_{hist_start}_{hist_end}.md",
            ],
        },
        "aggregate_comparison": {
            "command": [
                args.python_bin,
                "bin/evaluation/aggregate_perfect_model_comparisons.py",
                "--exp",
                args.exp,
                "--simu-test",
                args.simu,
                "--var",
                args.var,
                "--test-name",
                args.test_name,
                "--chunks-dir",
                str(comparison_output_dir / "chunks"),
                "--output-dir",
                str(comparison_output_dir),
            ],
            "expected": [comparison_output_dir / f"{combined_comparison_stem}.csv"],
            "cleanup": [
                comparison_output_dir / f"{combined_comparison_stem}.csv",
                comparison_output_dir / f"{combined_comparison_stem}.md",
                comparison_output_dir / f"perfect_model_predictions_vs_truth_{args.exp}_{metrics_test_name}.csv",
                comparison_output_dir / f"perfect_model_predictions_vs_truth_{args.exp}_{metrics_test_name}.md",
            ],
        },
        "plot_score_comparison": {
            "command": [
                args.python_bin,
                "bin/evaluation/plot_perfect_model_comparison.py",
                "--exp",
                args.exp,
                "--simu-test",
                args.simu,
                "--var",
                args.var,
                "--unit",
                args.validation_unit,
                "--input-csv",
                str(comparison_output_dir / f"{combined_comparison_stem}.csv"),
            ],
            "expected": [
                GRAPHS_DIR / "metrics" / args.exp / f"perfect_model_method_comparison_{args.exp}_{args.simu}.png",
            ],
            "cleanup": [
                GRAPHS_DIR / "metrics" / args.exp / f"perfect_model_method_comparison_{args.exp}_{args.simu}.png",
                GRAPHS_DIR / "metrics" / args.exp / f"perfect_model_method_comparison_{args.exp}_{args.simu}.pdf",
            ],
        },
        "plot_distribution": {
            "command": [
                args.python_bin,
                "bin/evaluation/plot_perfect_model_distribution_pdf.py",
                "--exp",
                args.exp,
                "--simu-test",
                args.simu,
                "--var",
                args.var,
                "--unit",
                args.validation_unit,
                "--sample-dir",
                str(eval_dataset_dir),
                "--raw-sample-dir",
                str(eval_dataset_dir),
                "--input-csv",
                str(comparison_output_dir / f"{combined_comparison_stem}.csv"),
                "--window",
                f"{hist_start}_{hist_end}",
            ],
            "expected": [
                GRAPHS_DIR / "metrics" / args.exp / f"perfect_model_distribution_pdf_{args.exp}_{args.simu}_{args.var}.png",
            ],
            "cleanup": [
                GRAPHS_DIR / "metrics" / args.exp / f"perfect_model_distribution_pdf_{args.exp}_{args.simu}_{args.var}.png",
                GRAPHS_DIR / "metrics" / args.exp / f"perfect_model_distribution_pdf_{args.exp}_{args.simu}_{args.var}.pdf",
                validation_output_dir / f"perfect_model_distribution_pdf_{args.exp}_{args.simu}_{args.var}.md",
            ],
        },
        "metrics_ml_day": {
            "command": [
                args.python_bin,
                "bin/evaluation/compute_prediction_file_metrics.py",
                "--exp",
                args.exp,
                "--test-name",
                args.test_name,
                "--simu-test",
                args.simu,
                "--var",
                args.var,
                "--startdate",
                hist_start,
                "--enddate",
                hist_end,
                "--sample-dir",
                str(eval_dataset_dir),
                "--prediction-path",
                str(prediction_path),
                "--frequency",
                "daily",
            ],
            "expected": [
                METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_mean_daily_{args.exp}_{metrics_test_name}.csv",
            ],
            "cleanup": [
                METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_daily_{args.exp}_{metrics_test_name}.npz",
                METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_mean_daily_{args.exp}_{metrics_test_name}.csv",
            ],
        },
        "metrics_ml_month": {
            "command": [
                args.python_bin,
                "bin/evaluation/compute_prediction_file_metrics.py",
                "--exp",
                args.exp,
                "--test-name",
                args.test_name,
                "--simu-test",
                args.simu,
                "--var",
                args.var,
                "--startdate",
                hist_start,
                "--enddate",
                hist_end,
                "--sample-dir",
                str(eval_dataset_dir),
                "--prediction-path",
                str(prediction_path),
                "--frequency",
                "monthly",
            ],
            "expected": [
                METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_mean_monthly_{args.exp}_{metrics_test_name}.csv",
            ],
            "cleanup": [
                METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_monthly_{args.exp}_{metrics_test_name}.npz",
                METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_mean_monthly_{args.exp}_{metrics_test_name}.csv",
            ],
        },
        "value_metrics": {
            "command": [
                args.python_bin,
                "bin/evaluation/compute_value_metrics.py",
                "--exp",
                args.exp,
                "--test-name",
                args.test_name,
                "--simu-test",
                args.simu,
                "--simu",
                args.simu,
                "--startdate",
                hist_start,
                "--enddate",
                hist_end,
            ],
            "expected": [get_value_metrics_output_path(args.exp, args.test_name, args.simu)],
            "cleanup": [get_value_metrics_output_path(args.exp, args.test_name, args.simu)],
        },
        "metrics_rcm_day": {
            "command": [
                args.python_bin,
                "bin/evaluation/compute_prediction_file_metrics.py",
                "--exp",
                args.exp,
                "--test-name",
                args.test_name,
                "--simu-test",
                args.simu,
                "--var",
                args.var,
                "--startdate",
                hist_start,
                "--enddate",
                hist_end,
                "--sample-dir",
                str(eval_dataset_dir),
                "--prediction-path",
                str(prediction_path),
                "--frequency",
                "daily",
                "--suffix",
                "_pp",
            ],
            "expected": [METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_mean_daily_{args.exp}_{metrics_test_name}_pp.csv"],
            "cleanup": [
                METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_daily_{args.exp}_{metrics_test_name}_pp.npz",
                METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_mean_daily_{args.exp}_{metrics_test_name}_pp.csv",
            ],
        },
        "metrics_rcm_month": {
            "command": [
                args.python_bin,
                "bin/evaluation/compute_prediction_file_metrics.py",
                "--exp",
                args.exp,
                "--test-name",
                args.test_name,
                "--simu-test",
                args.simu,
                "--var",
                args.var,
                "--startdate",
                hist_start,
                "--enddate",
                hist_end,
                "--sample-dir",
                str(eval_dataset_dir),
                "--prediction-path",
                str(prediction_path),
                "--frequency",
                "monthly",
                "--suffix",
                "_pp",
            ],
            "expected": [METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_mean_monthly_{args.exp}_{metrics_test_name}_pp.csv"],
            "cleanup": [
                METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_monthly_{args.exp}_{metrics_test_name}_pp.npz",
                METRICS_DIR / args.exp / "mean_metrics" / f"metrics_test_mean_monthly_{args.exp}_{metrics_test_name}_pp.csv",
            ],
        },
    }

    if args.var == "tas":
        step_table["plot_score_comparison"]["command"].extend(["--bias-tolerance", "0.10"])

    for step in steps:
        payload = step_table[step]
        run_step(
            name=step,
            command=payload["command"],
            expected_outputs=payload["expected"],
            cleanup_targets=payload["cleanup"],
            if_exists=args.if_exists,
            dry_run=args.dry_run,
        )

    print("[done] RCM perfect-model workflow finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
