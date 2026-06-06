#!/usr/bin/env python3
"""Plot perfect-model distribution PDFs for ML methods and pseudo-truth."""

from __future__ import annotations

import argparse
from pathlib import Path
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
    get_bias_corrected_netcdf_path,
    get_prediction_output_path,
    normalize_bc_tag,
)


MODEL_LABELS = {
    "bc_baseline": "BC baseline",
    "unet_outputnorm_perfect_model_rcm": "UNet + output norm",
    "unet_perfect_model_rcm": "UNet",
    "unet_rep3_perfect_model_rcm": "UNet replicate",
    "miniunet_perfect_model_rcm": "MiniUNet",
    "unet_seed2_perfect_model_rcm": "UNet cold replicate",
}

MODEL_COLORS = {
    "bc_baseline": "#2A9D8F",
    "unet_outputnorm_perfect_model_rcm": "#006D77",
    "unet_perfect_model_rcm": "#E76F51",
    "unet_rep3_perfect_model_rcm": "#457B9D",
    "miniunet_perfect_model_rcm": "#8D6E63",
    "unet_seed2_perfect_model_rcm": "#B56576",
}


REFERENCE_COLORS = {
    "truth": "#111111",
    "coarse_input": "#7A7A7A",
}


def model_label(model: str) -> str:
    if model.startswith("bc_baseline_") and model not in MODEL_LABELS:
        suffix = model[len("bc_baseline_") :].replace("_", " ")
        return f"BC baseline ({suffix})"
    return MODEL_LABELS.get(model, model)


def model_color(model: str) -> str:
    if model.startswith("bc_baseline_") and model not in MODEL_COLORS:
        return "#5B8E7D"
    return MODEL_COLORS.get(model, "#4A4A4A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="perfect_model_rcm")
    parser.add_argument("--simu-test", default="rcm")
    parser.add_argument("--var", default=None, help="Variable to read. Defaults to the experiment target variable.")
    parser.add_argument("--unit", default=None, help="Fallback unit if NetCDF metadata is missing.")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--raw-sample-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument(
        "--window",
        action="append",
        help="YYYYMMDD_YYYYMMDD window to plot. Repeat to compare several windows. Defaults to historical and late-century.",
    )
    return parser.parse_args()


def get_var_metadata(path: Path, var: str, fallback_unit: str | None) -> tuple[str, str]:
    with xr.open_dataset(path) as ds:
        attrs = dict(ds[var].attrs) if var in ds else {}
    label = attrs.get("long_name") or attrs.get("standard_name") or var
    unit = attrs.get("units") or fallback_unit or ""
    return str(label), str(unit)


def dates_from_window(window: str) -> pd.DatetimeIndex:
    start, end = window.split("_")
    return pd.date_range(start=start, end=end, freq="D")


def resolve_window_sample_dir(exp: str, simu_test: str, base_sample_dir: Path | None, window: str) -> Path:
    dates = dates_from_window(window)
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


def prediction_path(exp: str, simu_test: str, var: str, window: str, model: str) -> Path:
    start, end = window.split("_")
    test_name = f"{model}_{simu_test}"
    return get_prediction_output_path(exp, simu_test, var, start, end, test_name, ssp=CONFIG[exp].get("ssp"))


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
    bc_path = get_bias_corrected_netcdf_path(
        exp,
        simu_test,
        var,
        resolve_bc_period(window),
        ssp=CONFIG[exp].get("ssp"),
        bc_tag=normalize_bc_tag(bc_tag),
    )
    ds = xr.open_dataset(bc_path).sel(time=slice(pd.Timestamp(start_s), pd.Timestamp(end_s)))
    return reformat_as_target(
        ds,
        target_file=CONFIG[exp]["target_file"],
        domain=CONFIG[exp]["domain"],
        method="conservative_normed",
        mask=True,
        input_projection=None,
        reuse_weights=True,
    )


def update_range(current: tuple[float, float] | None, values: np.ndarray) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return current
    vmin = float(finite.min())
    vmax = float(finite.max())
    if current is None:
        return vmin, vmax
    return min(current[0], vmin), max(current[1], vmax)


def collect_range(
    sample_dir: Path,
    raw_sample_dir: Path,
    window: str,
    predictions: dict[str, Path | None],
    bc_tags: dict[str, str],
    var: str,
    exp: str,
    simu_test: str,
) -> tuple[float, float]:
    value_range: tuple[float, float] | None = None
    for date in dates_from_window(window):
        sample = np.load(sample_dir / f"sample_{date.strftime('%Y%m%d')}.npz")
        raw_sample = np.load(raw_sample_dir / f"sample_{date.strftime('%Y%m%d')}.npz")
        value_range = update_range(value_range, sample["y"][0])
        value_range = update_range(value_range, raw_sample["x"][1])
    for model, path in predictions.items():
        if model == "bc_baseline" or model.startswith("bc_baseline_"):
            with open_bc_on_target_grid_for_tag(exp, simu_test, var, window, bc_tags.get(model)) as ds:
                value_range = update_range(value_range, ds[var].values)
        else:
            with xr.open_dataset(path) as ds:
                value_range = update_range(value_range, ds[var].values)
    if value_range is None:
        raise ValueError(f"No finite values found for window {window}")
    span = value_range[1] - value_range[0]
    pad = span * 0.02 if span > 0 else 1.0
    return value_range[0] - pad, value_range[1] + pad


def add_hist(hist: np.ndarray, values: np.ndarray, edges: np.ndarray) -> None:
    finite = values[np.isfinite(values)]
    if finite.size:
        hist += np.histogram(finite, bins=edges)[0]


def density_from_hist(hist: np.ndarray, edges: np.ndarray) -> np.ndarray:
    widths = np.diff(edges)
    total = hist.sum()
    if total == 0:
        return np.full_like(hist, np.nan, dtype=float)
    return hist / (total * widths)


def write_markdown(path: Path, *, exp: str, simu_test: str, var: str, label: str, unit: str, windows: list[str], sample_dir: Path, models: list[str]) -> None:
    lines = [
        "# Perfect-Model Distribution PDF Diagnostic",
        "",
        f"- exp: `{exp}`",
        f"- simulation role: `{simu_test}`",
        f"- variable: `{var}`",
        f"- variable label: `{label}`",
        f"- unit: `{unit or 'not available in metadata'}`",
        f"- sample directory: `{sample_dir}`",
        f"- reference curves: native RCM pseudo-truth `y_{var}` and the raw degraded coarse-resolution predictor channel for `{var}`",
        f"- windows: `{', '.join(windows)}`",
        f"- ML methods: `{', '.join(models)}`",
        "",
        "This diagnostic is part of perfect-model validation: the ML curves should approach the pseudo-truth distribution more closely than the coarse-resolution input, without introducing a systematic shift in the distribution.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    var = args.var or CONFIG[args.exp]["target_vars"][0]
    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else METRICS_DIR
        / args.exp
        / "comparison_tables"
        / f"perfect_model_predictions_vs_truth_{args.exp}_combined_{args.simu_test}.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else GRAPHS_DIR / "metrics" / args.exp
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = METRICS_DIR / args.exp / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(input_csv)
    discovered = list(summary["model"].unique())
    bc_tags: dict[str, str] = {}
    if "bc_tag" in summary.columns:
        for model in discovered:
            rows = summary[summary["model"] == model]
            if rows.empty:
                continue
            tag = normalize_bc_tag(str(rows["bc_tag"].fillna("").iloc[0]))
            if tag:
                bc_tags[model] = tag
    models = [model for model in MODEL_LABELS if model in set(discovered)]
    models += [model for model in discovered if model not in models]
    windows = args.window or ["20000101_20141231", "20900101_21001231"]
    raw_sample_dir = Path(args.raw_sample_dir) if args.raw_sample_dir else Path(CONFIG[args.exp]["dataset"])

    first_model = next((model for model in models if not model.startswith("bc_baseline")), None)
    if first_model is None:
        raise ValueError("No ML model found in the summary table.")
    first_path = prediction_path(args.exp, args.simu_test, var, windows[0], first_model)
    var_label, unit = get_var_metadata(first_path, var, args.unit)
    if "var_label" in summary and summary["var_label"].notna().any():
        var_label = str(summary["var_label"].dropna().iloc[0])
    if "unit" in summary and summary["unit"].notna().any() and str(summary["unit"].dropna().iloc[0]):
        unit = str(summary["unit"].dropna().iloc[0])
    unit_label = f" [{unit}]" if unit else ""

    fig, axes = plt.subplots(1, len(windows), figsize=(8.5 * len(windows), 5.2), squeeze=False, constrained_layout=True)
    fig.suptitle(f"Perfect-model probability density comparison: {var_label}", fontsize=15, fontweight="bold")

    for ax, window in zip(axes[0], windows):
        predictions = {
            model: None if model == "bc_baseline" or model.startswith("bc_baseline_") else prediction_path(args.exp, args.simu_test, var, window, model)
            for model in models
        }
        window_sample_dir = resolve_window_sample_dir(args.exp, args.simu_test, sample_dir, window)
        value_min, value_max = collect_range(
            window_sample_dir,
            raw_sample_dir,
            window,
            predictions,
            bc_tags,
            var,
            args.exp,
            args.simu_test,
        )
        edges = np.linspace(value_min, value_max, args.bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        histograms = {
            "truth": np.zeros(args.bins, dtype=np.float64),
            "coarse_input": np.zeros(args.bins, dtype=np.float64),
            **{model: np.zeros(args.bins, dtype=np.float64) for model in models},
        }

        dates = dates_from_window(window)
        for date in dates:
            sample = np.load(window_sample_dir / f"sample_{date.strftime('%Y%m%d')}.npz")
            raw_sample = np.load(raw_sample_dir / f"sample_{date.strftime('%Y%m%d')}.npz")
            add_hist(histograms["truth"], sample["y"][0], edges)
            add_hist(histograms["coarse_input"], raw_sample["x"][1], edges)

        for model, path in predictions.items():
            if model == "bc_baseline" or model.startswith("bc_baseline_"):
                with open_bc_on_target_grid_for_tag(args.exp, args.simu_test, var, window, bc_tags.get(model)) as ds:
                    for i in range(ds.sizes["time"]):
                        add_hist(histograms[model], ds[var].isel(time=i).values, edges)
            else:
                with xr.open_dataset(path) as ds:
                    for i in range(ds.sizes["time"]):
                        add_hist(histograms[model], ds[var].isel(time=i).values, edges)

        ax.plot(centers, density_from_hist(histograms["truth"], edges), lw=2.8, color=REFERENCE_COLORS["truth"], label="Native RCM pseudo-truth")
        ax.plot(
            centers,
            density_from_hist(histograms["coarse_input"], edges),
            lw=2.2,
            ls="--",
            color=REFERENCE_COLORS["coarse_input"],
            label="RCM coarse-resolution input",
        )
        for model in models:
            ax.plot(centers, density_from_hist(histograms[model], edges), lw=1.9, color=model_color(model), alpha=0.9, label=model_label(model))

        start, end = window.split("_")
        ax.set_title(f"{start[:4]}-{end[:4]}")
        ax.set_xlabel(f"{var_label}{unit_label}")
        ax.set_ylabel("Probability density")
        ax.grid(alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)

    stem = f"perfect_model_distribution_pdf_{args.exp}_{args.simu_test}_{var}"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    md = validation_dir / f"{stem}.md"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    write_markdown(
        md,
        exp=args.exp,
        simu_test=args.simu_test,
        var=var,
        label=var_label,
        unit=unit,
        windows=windows,
        sample_dir=sample_dir or Path(CONFIG[args.exp]["dataset"]),
        models=models,
    )
    print(f"png={png}")
    print(f"pdf={pdf}")
    print(f"validation_md={md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
