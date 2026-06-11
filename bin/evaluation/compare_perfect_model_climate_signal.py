#!/usr/bin/env python3
"""Compare perfect-model climate-change signals for raw, BC, and ML methods."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
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
    get_input_channel_index,
    get_target_channel_index,
    get_bias_corrected_netcdf_path,
    normalize_bc_tag,
)


MODEL_LABELS = {
    "raw_input": "RCM coarse-resolution input",
    "bc_baseline": "BC baseline",
    "bc_baseline_sbck_cdft": "SBCK CDFt baseline",
    "unet_outputnorm_perfect_model_rcm": "UNet + output norm",
    "unet_perfect_model_rcm": "UNet",
    "unet_rep3_perfect_model_rcm": "UNet replicate",
    "miniunet_perfect_model_rcm": "MiniUNet",
    "unet_seed2_perfect_model_rcm": "UNet cold replicate",
}

MODEL_COLORS = {
    "raw_input": "#7A7A7A",
    "bc_baseline": "#2A9D8F",
    "bc_baseline_sbck_cdft": "#5B8E7D",
    "unet_outputnorm_perfect_model_rcm": "#006D77",
    "unet_perfect_model_rcm": "#E76F51",
    "unet_rep3_perfect_model_rcm": "#457B9D",
    "miniunet_perfect_model_rcm": "#8D6E63",
    "unet_seed2_perfect_model_rcm": "#B56576",
}


@dataclass
class SignalSummary:
    method: str
    reference_mean: float
    future_mean: float
    signal_mean: float
    truth_signal_mean: float
    signal_bias_mean: float
    signal_rmse: float
    signal_corr: float
    method_signal_std: float
    truth_signal_std: float


def model_label(model: str) -> str:
    if model.startswith("bc_baseline_") and model not in MODEL_LABELS:
        suffix = model[len("bc_baseline_") :].replace("_", " ")
        return f"BC baseline ({suffix})"
    return MODEL_LABELS.get(model, model)


def model_color(model: str) -> str:
    if model.startswith("bc_baseline_") and model not in MODEL_COLORS:
        return "#5B8E7D"
    return MODEL_COLORS.get(model, "#4A4A4A")


def resolve_channel_indices(exp: str, var: str) -> tuple[int, int]:
    return get_input_channel_index(exp, var), get_target_channel_index(exp, var)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="perfect_model_rcm")
    parser.add_argument("--simu-test", default="rcm")
    parser.add_argument("--var", default="tas")
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--raw-sample-dir", default=None)
    parser.add_argument("--input-csv", default=None, help="Optional perfect-model combined comparison CSV used to auto-discover ML methods.")
    parser.add_argument("--model", action="append", default=None, help="ML method to include. Repeat to select several methods.")
    parser.add_argument("--reference-window", default="19810101_20101231")
    parser.add_argument("--future-window", default="20800101_21001231")
    parser.add_argument("--unit", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot-dir", default=None)
    return parser.parse_args()


def resolve_models(args: argparse.Namespace) -> list[str]:
    if args.model:
        return args.model
    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else METRICS_DIR / args.exp / "comparison_tables" / f"perfect_model_predictions_vs_truth_{args.exp}_combined_{args.simu_test}.csv"
    )
    df = pd.read_csv(input_csv)
    models = [model for model in df["model"].unique() if model != "bc_baseline" and not model.startswith("bc_baseline_")]
    return [model for model in MODEL_LABELS if model in models and model not in {"raw_input", "bc_baseline"}] + [
        model for model in models if model not in MODEL_LABELS
    ]


def resolve_bc_methods(args: argparse.Namespace) -> list[tuple[str, str]]:
    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else METRICS_DIR / args.exp / "comparison_tables" / f"perfect_model_predictions_vs_truth_{args.exp}_combined_{args.simu_test}.csv"
    )
    df = pd.read_csv(input_csv)
    bc_models = [model for model in df["model"].unique() if model == "bc_baseline" or str(model).startswith("bc_baseline_")]
    resolved: list[tuple[str, str]] = []
    for model in bc_models:
        rows = df[df["model"] == model]
        tag = ""
        if "bc_tag" in rows.columns and not rows.empty:
            tag = normalize_bc_tag(str(rows["bc_tag"].fillna("").iloc[0]))
        resolved.append((str(model), tag))
    return resolved


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
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No sample directory found for window {window}. Searched: {searched}")


def resolve_bc_period(window: str) -> str:
    start_s, end_s = window.split("_")
    start = pd.Timestamp(start_s)
    end = pd.Timestamp(end_s)
    if start >= DATES_BC_TRAIN_HIST[0] and end <= DATES_BC_TRAIN_HIST[-1]:
        return "train_hist"
    if start >= DATES_BC_TEST_HIST[0] and end <= DATES_BC_TEST_HIST[-1]:
        return "test_hist"
    return "test_future"


def open_bc_on_target_grid(exp: str, simu_test: str, var: str, window: str) -> xr.Dataset:
    return open_bc_on_target_grid_for_tag(exp, simu_test, var, window, bc_tag=None)


def open_bc_on_target_grid_for_tag(exp: str, simu_test: str, var: str, window: str, bc_tag: str | None) -> xr.Dataset:
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
    escaped_exp = re.escape(exp)
    escaped_model = re.escape(model)
    escaped_simu = re.escape(simu_test)
    return re.compile(rf"_(\d{{8}})_(\d{{8}})_{escaped_exp}_{escaped_model}_{escaped_simu}\.nc$")


def open_prediction_window(exp: str, model: str, simu_test: str, window: str, var: str) -> xr.Dataset:
    start_s, end_s = window.split("_")
    start = pd.Timestamp(start_s)
    end = pd.Timestamp(end_s)
    pattern = prediction_pattern(exp, model, simu_test)
    matched: list[tuple[pd.Timestamp, pd.Timestamp, Path]] = []
    for path in sorted(Path(PREDICTION_DIR).glob(f"{var}_day_*_{exp}_{model}_{simu_test}.nc")):
        m = pattern.search(path.name)
        if not m:
            continue
        chunk_start = pd.Timestamp(m.group(1))
        chunk_end = pd.Timestamp(m.group(2))
        if chunk_end < start or chunk_start > end:
            continue
        matched.append((chunk_start, chunk_end, path))
    if not matched:
        raise FileNotFoundError(f"No prediction files found for {model} over {window}")
    datasets = [xr.open_dataset(path).sel(time=slice(start, end)) for _, _, path in matched]
    if len(datasets) == 1:
        return datasets[0]
    combined = xr.concat(datasets, dim="time").sortby("time")
    for ds in datasets:
        ds.close()
    return combined


def time_mean_from_samples(sample_dir: Path, dates: pd.DatetimeIndex, key: str, channel_index: int) -> np.ndarray:
    acc = None
    count = 0
    for date in dates:
        sample = np.load(sample_dir / f"sample_{date.strftime('%Y%m%d')}.npz")
        values = sample[key][channel_index].astype(np.float64, copy=False)
        if acc is None:
            acc = np.zeros_like(values, dtype=np.float64)
        acc += values
        count += 1
    if acc is None or count == 0:
        raise ValueError(f"No samples found for {key} over {dates[0]}..{dates[-1]}")
    return acc / count


def time_mean_from_prediction(ds: xr.Dataset, var: str) -> np.ndarray:
    return ds[var].mean(dim="time").values.astype(np.float64, copy=False)


def summarize_signal(method: str, method_signal: np.ndarray, truth_signal: np.ndarray, reference_mean: np.ndarray, future_mean: np.ndarray) -> SignalSummary:
    mask = np.isfinite(method_signal) & np.isfinite(truth_signal)
    mv = method_signal[mask].astype(np.float64, copy=False)
    tv = truth_signal[mask].astype(np.float64, copy=False)
    diff = mv - tv
    mean_m = float(np.nanmean(mv))
    mean_t = float(np.nanmean(tv))
    var_m = float(np.nanvar(mv))
    var_t = float(np.nanvar(tv))
    cov = float(np.nanmean((mv - mean_m) * (tv - mean_t)))
    corr = float(cov / np.sqrt(var_m * var_t)) if var_m > 0 and var_t > 0 else np.nan
    return SignalSummary(
        method=method,
        reference_mean=float(np.nanmean(reference_mean)),
        future_mean=float(np.nanmean(future_mean)),
        signal_mean=mean_m,
        truth_signal_mean=mean_t,
        signal_bias_mean=float(np.nanmean(diff)),
        signal_rmse=float(np.sqrt(np.nanmean(diff**2))),
        signal_corr=corr,
        method_signal_std=float(np.nanstd(mv)),
        truth_signal_std=float(np.nanstd(tv)),
    )


def window_year_label(window: str) -> str:
    start, end = window.split("_")
    return f"{start[:4]}-{end[:4]}"


def write_markdown(path: Path, rows: pd.DataFrame, exp: str, simu_test: str, var: str, unit: str, reference_window: str, future_window: str) -> None:
    cols = list(rows.columns)
    table_rows = [cols, ["---"] * len(cols)]
    for _, row in rows.iterrows():
        rendered = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{value:.6f}" if np.isfinite(value) else "nan")
            else:
                rendered.append(str(value))
        table_rows.append(rendered)
    table = "\n".join("| " + " | ".join(r) + " |" for r in table_rows)
    path.write_text(
        "\n".join(
            [
                "# Perfect-Model Climate-Change Signal Comparison",
                "",
                f"- exp: `{exp}`",
                f"- simulation role: `{simu_test}`",
                f"- variable: `{var}`",
                f"- unit: `{unit or 'not available'}`",
                f"- reference window: `{reference_window}`",
                f"- future window: `{future_window}`",
                "- methods include the raw coarse-resolution RCM input, the configured BC baseline(s), and the selected ML methods.",
                "- `signal_*` metrics compare each method's change field against the native RCM pseudo-truth change field.",
                "",
                table,
                "",
            ]
        )
    )


def plot_rows(rows: pd.DataFrame, output_png: Path, output_pdf: Path, unit: str, reference_window: str, future_window: str) -> None:
    order = rows["method"].tolist()
    labels = [model_label(method) for method in order]
    colors = [model_color(method) for method in order]
    y = np.arange(len(order))
    unit_suffix = f" [{unit}]" if unit else ""
    rows = rows.copy()
    rows["signal_std_bias"] = rows["method_signal_std"] - rows["truth_signal_std"]
    rows["signal_corr_deficit"] = 1.0 - rows["signal_corr"]
    truth_mean = float(rows["truth_signal_mean"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(
        f"Perfect-model climate-change signal: {window_year_label(reference_window)} to {window_year_label(future_window)}",
        fontsize=15,
        fontweight="bold",
    )

    axes[0, 0].barh(y, rows["signal_bias_mean"], color=colors)
    axes[0, 0].axvline(0, color="#111111", lw=1.2)
    axes[0, 0].set_title(f"Mean warming error vs pseudo-truth ({truth_mean:.2f}{unit_suffix})")
    axes[0, 0].set_xlabel(f"Method signal - truth signal{unit_suffix}")
    axes[0, 0].set_yticks(y, labels)
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis="x", alpha=0.25)

    axes[0, 1].barh(y, rows["signal_rmse"], color=colors)
    axes[0, 1].set_title("Signal-field RMSE vs pseudo-truth")
    axes[0, 1].set_xlabel(f"RMSE{unit_suffix}")
    axes[0, 1].set_yticks(y, labels)
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis="x", alpha=0.25)

    axes[1, 0].barh(y, rows["signal_corr_deficit"], color=colors)
    axes[1, 0].set_title("Spatial-correlation deficit")
    axes[1, 0].set_xlabel("1 - correlation")
    axes[1, 0].set_yticks(y, labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis="x", alpha=0.25)

    axes[1, 1].barh(y, rows["signal_std_bias"], color=colors)
    axes[1, 1].axvline(0, color="#111111", lw=1.2)
    axes[1, 1].set_title("Spatial variability of the signal")
    axes[1, 1].set_xlabel(f"Method signal std - truth signal std{unit_suffix}")
    axes[1, 1].set_yticks(y, labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis="x", alpha=0.25)

    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    raw_sample_dir = Path(args.raw_sample_dir) if args.raw_sample_dir else Path(CONFIG[args.exp]["dataset"])
    output_dir = Path(args.output_dir) if args.output_dir else METRICS_DIR / args.exp / "comparison_tables"
    plot_dir = Path(args.plot_dir) if args.plot_dir else GRAPHS_DIR / "metrics" / args.exp
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    input_channel_index, target_channel_index = resolve_channel_indices(args.exp, args.var)
    models = resolve_models(args)
    bc_methods = resolve_bc_methods(args)
    methods = ["raw_input"] + [label for label, _ in bc_methods] + models

    ref_dates = window_dates(args.reference_window)
    fut_dates = window_dates(args.future_window)
    ref_sample_dir = resolve_window_sample_dir(args.exp, args.simu_test, sample_dir, args.reference_window)
    fut_sample_dir = resolve_window_sample_dir(args.exp, args.simu_test, sample_dir, args.future_window)
    truth_ref = time_mean_from_samples(ref_sample_dir, ref_dates, "y", target_channel_index)
    truth_fut = time_mean_from_samples(fut_sample_dir, fut_dates, "y", target_channel_index)
    truth_signal = truth_fut - truth_ref

    raw_ref = time_mean_from_samples(resolve_window_sample_dir(args.exp, args.simu_test, raw_sample_dir, args.reference_window), ref_dates, "x", input_channel_index)
    raw_fut = time_mean_from_samples(resolve_window_sample_dir(args.exp, args.simu_test, raw_sample_dir, args.future_window), fut_dates, "x", input_channel_index)
    rows = [
        summarize_signal("raw_input", raw_fut - raw_ref, truth_signal, raw_ref, raw_fut),
    ]

    for bc_model, bc_tag in bc_methods:
        with open_bc_on_target_grid_for_tag(args.exp, args.simu_test, args.var, args.reference_window, bc_tag) as bc_ref_ds:
            bc_ref = time_mean_from_prediction(bc_ref_ds, args.var)
        with open_bc_on_target_grid_for_tag(args.exp, args.simu_test, args.var, args.future_window, bc_tag) as bc_fut_ds:
            bc_fut = time_mean_from_prediction(bc_fut_ds, args.var)
        rows.append(summarize_signal(bc_model, bc_fut - bc_ref, truth_signal, bc_ref, bc_fut))

    unit = args.unit or ""
    for model in models:
        with open_prediction_window(args.exp, model, args.simu_test, args.reference_window, args.var) as ref_ds:
            if not unit and args.var in ref_ds and ref_ds[args.var].attrs.get("units"):
                unit = str(ref_ds[args.var].attrs.get("units"))
            ref_mean = time_mean_from_prediction(ref_ds, args.var)
        with open_prediction_window(args.exp, model, args.simu_test, args.future_window, args.var) as fut_ds:
            fut_mean = time_mean_from_prediction(fut_ds, args.var)
        rows.append(summarize_signal(model, fut_mean - ref_mean, truth_signal, ref_mean, fut_mean))

    df = pd.DataFrame([row.__dict__ for row in rows])
    df.insert(1, "label", df["method"].map(model_label))

    stem = f"perfect_model_climate_signal_{args.exp}_{args.simu_test}_{args.reference_window}_vs_{args.future_window}"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    png_path = plot_dir / f"{stem}.png"
    pdf_path = plot_dir / f"{stem}.pdf"
    df.to_csv(csv_path, index=False)
    write_markdown(md_path, df, args.exp, args.simu_test, args.var, unit, args.reference_window, args.future_window)
    plot_rows(df, png_path, pdf_path, unit, args.reference_window, args.future_window)
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    print(f"png={png_path}")
    print(f"pdf={pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
