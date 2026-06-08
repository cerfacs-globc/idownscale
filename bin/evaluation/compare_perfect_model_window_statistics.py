#!/usr/bin/env python3
"""Compare perfect-model windows across raw input, BC baselines, and ML methods."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(".")

from iriscc.datautils import reformat_as_target
from iriscc.settings import (
    CONFIG,
    DATES_BC_TEST_HIST,
    DATES_BC_TRAIN_HIST,
    GRAPHS_DIR,
    METRICS_DIR,
    PREDICTION_DIR,
    get_bias_corrected_netcdf_path,
    normalize_bc_tag,
)


MODEL_LABELS = {
    "raw_input": "RCM coarse-resolution input",
    "bc_baseline": "BC baseline",
    "unet_outputnorm_perfect_model_rcm": "UNet + output norm",
    "unet_perfect_model_rcm": "UNet",
    "unet_rep3_perfect_model_rcm": "UNet replicate",
    "miniunet_perfect_model_rcm": "MiniUNet",
    "unet_seed2_perfect_model_rcm": "UNet cold replicate",
}


DEFAULT_WINDOWS = [
    "19810101_20101231",
    "20000101_20141231",
    "20150101_20291231",
    "20300101_20441231",
    "20450101_20591231",
    "20600101_20741231",
    "20750101_20891231",
    "20900101_21001231",
]


def model_label(model: str) -> str:
    if model.startswith("bc_baseline_") and model not in MODEL_LABELS:
        suffix = model[len("bc_baseline_") :].replace("_", " ")
        return f"BC baseline ({suffix})"
    return MODEL_LABELS.get(model, model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="perfect_model_rcm")
    parser.add_argument("--simu-test", default="rcm")
    parser.add_argument("--var", default="tas")
    parser.add_argument("--sample-dir", default=None, help="Target-bearing sample directory used for truth y.")
    parser.add_argument("--raw-sample-dir", default=None, help="Raw degraded-input sample directory used for the raw-input baseline.")
    parser.add_argument("--input-csv", default=None, help="Combined perfect-model summary CSV used to auto-discover methods.")
    parser.add_argument("--model", action="append", default=None, help="ML method to include. Repeat to select several methods.")
    parser.add_argument(
        "--window",
        action="append",
        default=None,
        help="YYYYMMDD_YYYYMMDD window to include. Repeat to select several windows.",
    )
    parser.add_argument("--unit", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def summary_csv_path(args: argparse.Namespace) -> Path:
    return (
        Path(args.input_csv)
        if args.input_csv
        else METRICS_DIR / args.exp / "comparison_tables" / f"perfect_model_predictions_vs_truth_{args.exp}_combined_{args.simu_test}.csv"
    )


def resolve_models(args: argparse.Namespace) -> list[str]:
    if args.model:
        return args.model
    df = pd.read_csv(summary_csv_path(args))
    models = [model for model in df["model"].unique() if model != "bc_baseline" and not str(model).startswith("bc_baseline_")]
    return [model for model in MODEL_LABELS if model in models and model not in {"raw_input", "bc_baseline"}] + [
        model for model in models if model not in MODEL_LABELS
    ]


def resolve_bc_methods(args: argparse.Namespace) -> list[tuple[str, str]]:
    df = pd.read_csv(summary_csv_path(args))
    models = [model for model in df["model"].unique() if model == "bc_baseline" or str(model).startswith("bc_baseline_")]
    resolved: list[tuple[str, str]] = []
    for model in models:
        rows = df[df["model"] == model]
        tag = ""
        if "bc_tag" in rows.columns and not rows.empty:
            tag = normalize_bc_tag(str(rows["bc_tag"].fillna("").iloc[0]))
        resolved.append((str(model), tag))
    return resolved


def resolve_channel_indices(exp: str, var: str) -> tuple[int, int]:
    input_vars = CONFIG[exp].get("input_vars", [])
    target_vars = CONFIG[exp].get("target_vars", [])
    input_index = input_vars.index(var) if var in input_vars else max(len(input_vars) - 1, 0)
    target_index = target_vars.index(var) if var in target_vars else 0
    return input_index, target_index


def get_var_metadata(path: Path, var: str, fallback_unit: str | None) -> tuple[str, str]:
    with xr.open_dataset(path) as ds:
        attrs = dict(ds[var].attrs) if var in ds else {}
    label = attrs.get("long_name") or attrs.get("standard_name") or var
    unit = attrs.get("units") or fallback_unit or ""
    return str(label), str(unit)


def window_dates(window: str) -> pd.DatetimeIndex:
    start, end = window.split("_")
    return pd.date_range(start=start, end=end, freq="D")


def resolve_window_sample_dir(exp: str, simu_test: str, base_sample_dir: Path | None, window: str) -> Path:
    dates = window_dates(window)
    candidates: list[Path] = []
    if base_sample_dir is not None:
        candidates.append(base_sample_dir)
    for candidate in (
        Path(CONFIG[exp]["dataset"]),
        Path(CONFIG[exp]["dataset"]).parent / f"dataset_{exp}_test_{simu_test}_bc",
        Path(CONFIG[exp]["dataset"]).parent / f"dataset_{exp}_validation_windows_{simu_test}_bc",
    ):
        if candidate not in candidates:
            candidates.append(candidate)

    first_name = f"sample_{dates[0].strftime('%Y%m%d')}.npz"
    last_name = f"sample_{dates[-1].strftime('%Y%m%d')}.npz"
    for candidate in candidates:
        if candidate.exists() and (candidate / first_name).exists() and (candidate / last_name).exists():
            return candidate
    raise FileNotFoundError(f"No sample directory found for window {window}. Searched: {', '.join(str(p) for p in candidates)}")


def resolve_bc_period(window: str) -> str:
    start_s, end_s = window.split("_")
    start = pd.Timestamp(start_s)
    end = pd.Timestamp(end_s)
    if start >= DATES_BC_TRAIN_HIST[0] and end <= DATES_BC_TRAIN_HIST[-1]:
        return "train_hist"
    if start >= DATES_BC_TEST_HIST[0] and end <= DATES_BC_TEST_HIST[-1]:
        return "test_hist"
    return "test_future"


def open_bc_on_target_grid(exp: str, simu_test: str, var: str, window: str, bc_tag: str | None) -> xr.Dataset:
    start_s, end_s = window.split("_")
    start = pd.Timestamp(start_s)
    end = pd.Timestamp(end_s)
    periods = []
    train_start, train_end = DATES_BC_TRAIN_HIST[0], DATES_BC_TRAIN_HIST[-1]
    hist_start, hist_end = DATES_BC_TEST_HIST[0], DATES_BC_TEST_HIST[-1]
    if start <= train_end and end >= train_start:
        periods.append(("train_hist", max(start, train_start), min(end, train_end)))
    if start <= hist_end and end >= hist_start:
        periods.append(("test_hist", max(start, hist_start), min(end, hist_end)))
    if end > hist_end:
        periods.append(("test_future", max(start, hist_end + pd.Timedelta(days=1)), end))
    datasets = []
    for period, sub_start, sub_end in periods:
        bc_path = get_bias_corrected_netcdf_path(
            exp,
            simu_test,
            var,
            period,
            ssp=CONFIG[exp].get("ssp"),
            bc_tag=normalize_bc_tag(bc_tag),
        )
        datasets.append(xr.open_dataset(bc_path).sel(time=slice(sub_start, sub_end)))
    if not datasets:
        raise ValueError(f"No BC period overlaps requested window {window}")
    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = xr.concat(datasets, dim="time").sortby("time")
        for item in datasets:
            item.close()
    return reformat_as_target(
        ds,
        target_file=CONFIG[exp]["target_file"],
        domain=CONFIG[exp]["domain"],
        method="conservative_normed",
        mask=True,
        input_projection=None,
        reuse_weights=True,
    )


def prediction_pattern(exp: str, model: str, simu_test: str) -> re.Pattern[str]:
    return re.compile(rf"_(\d{{8}})_(\d{{8}})_{re.escape(exp)}_{re.escape(model)}_{re.escape(simu_test)}\.nc$")


def open_prediction_window(exp: str, model: str, simu_test: str, window: str, var: str) -> xr.Dataset:
    start_s, end_s = window.split("_")
    start = pd.Timestamp(start_s)
    end = pd.Timestamp(end_s)
    pattern = prediction_pattern(exp, model, simu_test)
    matched: list[Path] = []
    for path in sorted(Path(PREDICTION_DIR).glob(f"{var}_day_*_{exp}_{model}_{simu_test}.nc")):
        m = pattern.search(path.name)
        if not m:
            continue
        chunk_start = pd.Timestamp(m.group(1))
        chunk_end = pd.Timestamp(m.group(2))
        if chunk_end < start or chunk_start > end:
            continue
        matched.append(path)
    if not matched:
        raise FileNotFoundError(f"No prediction files found for {model} over {window}")
    datasets = [xr.open_dataset(path).sel(time=slice(start, end)) for path in matched]
    if len(datasets) == 1:
        return datasets[0]
    combined = xr.concat(datasets, dim="time").sortby("time")
    _, unique_idx = np.unique(combined.time.values, return_index=True)
    combined = combined.isel(time=np.sort(unique_idx))
    combined = combined.sel(time=slice(start, end))
    for ds in datasets:
        ds.close()
    return combined


def load_truth_array(sample_dir: Path, dates: pd.DatetimeIndex, target_channel_index: int) -> np.ndarray:
    arrays = [np.load(sample_dir / f"sample_{date.strftime('%Y%m%d')}.npz")["y"][target_channel_index].astype(np.float32) for date in dates]
    return np.stack(arrays, axis=0)


def load_raw_array(sample_dir: Path, dates: pd.DatetimeIndex, input_channel_index: int) -> np.ndarray:
    arrays = [np.load(sample_dir / f"sample_{date.strftime('%Y%m%d')}.npz")["x"][input_channel_index].astype(np.float32) for date in dates]
    return np.stack(arrays, axis=0)


def yearly_area_mean_std(values: np.ndarray, dates: pd.DatetimeIndex) -> float:
    area_mean = np.nanmean(values, axis=(1, 2))
    frame = pd.DataFrame({"date": dates, "value": area_mean})
    frame["year"] = frame["date"].dt.year
    annual = frame.groupby("year", observed=False)["value"].mean().to_numpy()
    return float(np.nanstd(annual)) if annual.size else np.nan


def summarize_window(method: str, values: np.ndarray, truth: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, object]:
    diff = values - truth
    metrics = {
        "model": method,
        "label": model_label(method),
        "days": int(values.shape[0]),
        "mean_bias": float(np.nanmean(diff)),
        "rmse": float(np.sqrt(np.nanmean(diff**2))),
        "mean_value": float(np.nanmean(values)),
        "truth_mean": float(np.nanmean(truth)),
        "mean_abs_error": float(np.nanmean(np.abs(diff))),
    }
    if values.shape[0] > 1:
        method_dt = np.diff(values, axis=0)
        truth_dt = np.diff(truth, axis=0)
        metrics["day_to_day_variability_bias"] = float(np.nanmean(np.abs(method_dt)) - np.nanmean(np.abs(truth_dt)))
    else:
        metrics["day_to_day_variability_bias"] = np.nan

    method_spatial_std = np.nanstd(values, axis=(1, 2))
    truth_spatial_std = np.nanstd(truth, axis=(1, 2))
    metrics["spatial_variability_bias"] = float(np.nanmean(method_spatial_std - truth_spatial_std))

    metrics["annual_variability_bias"] = yearly_area_mean_std(values, dates) - yearly_area_mean_std(truth, dates)
    return metrics


def write_markdown(path: Path, df: pd.DataFrame, *, exp: str, simu_test: str, var: str, var_label: str, unit: str, windows: list[str]) -> None:
    cols = list(df.columns)
    rows = [cols, ["---"] * len(cols)]
    for _, row in df.iterrows():
        rendered = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{value:.6f}" if np.isfinite(value) else "nan")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    table = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    path.write_text(
        "\n".join(
            [
                "# Perfect-model window-by-window diagnostics",
                "",
                f"- experiment: `{exp}`",
                f"- simulation role: `{simu_test}`",
                f"- variable: `{var}`",
                f"- variable label: `{var_label}`",
                f"- unit: `{unit or 'not available'}`",
                f"- windows: `{', '.join(windows)}`",
                "",
                "The table below complements RMSE/bias with three additional diagnostics that are closer to the real downscaling validation workflow:",
                "",
                "- `day_to_day_variability_bias`: mean absolute daily increment minus pseudo-truth",
                "- `spatial_variability_bias`: mean daily spatial standard deviation minus pseudo-truth",
                "- `annual_variability_bias`: standard deviation of annual mean area-average temperature minus pseudo-truth",
                "",
                table,
                "",
            ]
        )
    )


def main() -> int:
    args = parse_args()
    windows = args.window or list(DEFAULT_WINDOWS)
    input_channel_index, target_channel_index = resolve_channel_indices(args.exp, args.var)
    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    raw_sample_dir = Path(args.raw_sample_dir) if args.raw_sample_dir else Path(CONFIG[args.exp]["dataset"])
    output_dir = Path(args.output_dir) if args.output_dir else METRICS_DIR / args.exp / "comparison_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = resolve_models(args)
    bc_methods = resolve_bc_methods(args)
    rows: list[dict[str, object]] = []
    var_label = args.var
    unit = args.unit or ""

    summary_path = summary_csv_path(args)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "var_label" in summary.columns and summary["var_label"].notna().any():
            var_label = str(summary["var_label"].dropna().iloc[0])
        if "unit" in summary.columns and summary["unit"].notna().any() and str(summary["unit"].dropna().iloc[0]):
            unit = str(summary["unit"].dropna().iloc[0])

    for window in windows:
        dates = window_dates(window)
        truth_sample_dir = resolve_window_sample_dir(args.exp, args.simu_test, sample_dir, window)
        raw_window_dir = resolve_window_sample_dir(args.exp, args.simu_test, raw_sample_dir, window)
        truth = load_truth_array(truth_sample_dir, dates, target_channel_index)
        raw = load_raw_array(raw_window_dir, dates, input_channel_index)

        raw_row = summarize_window("raw_input", raw, truth, dates)
        raw_row["window"] = window
        rows.append(raw_row)

        for bc_model, bc_tag in bc_methods:
            with open_bc_on_target_grid(args.exp, args.simu_test, args.var, window, bc_tag) as ds_bc:
                bc_values = ds_bc[args.var].values.astype(np.float32, copy=False)
            bc_row = summarize_window(bc_model, bc_values, truth, dates)
            bc_row["window"] = window
            rows.append(bc_row)

        for model in models:
            with open_prediction_window(args.exp, model, args.simu_test, window, args.var) as ds_pred:
                pred_values = ds_pred[args.var].values.astype(np.float32, copy=False)
            model_row = summarize_window(model, pred_values, truth, dates)
            model_row["window"] = window
            rows.append(model_row)

    df = pd.DataFrame(rows)
    stem = f"perfect_model_window_statistics_{args.exp}_{args.simu_test}"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    write_markdown(md_path, df, exp=args.exp, simu_test=args.simu_test, var=args.var, var_label=var_label, unit=unit, windows=windows)
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
