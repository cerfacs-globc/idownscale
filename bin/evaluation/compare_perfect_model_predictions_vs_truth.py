#!/usr/bin/env python3
"""Compare perfect-model predictions and degraded inputs against pseudo-truth y."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(".")

from iriscc.settings import CONFIG, METRICS_DIR, get_prediction_output_path


def resolve_channel_indices(exp: str, var: str) -> tuple[int, int]:
    input_vars = CONFIG[exp].get("input_vars", [])
    target_vars = CONFIG[exp].get("target_vars", [])
    input_index = input_vars.index(var) if var in input_vars else max(len(input_vars) - 1, 0)
    target_index = target_vars.index(var) if var in target_vars else 0
    return input_index, target_index


def get_prediction_metadata(prediction_path: Path, var: str, fallback_unit: str | None) -> dict[str, str]:
    with xr.open_dataset(prediction_path) as ds:
        attrs = dict(ds[var].attrs) if var in ds else {}
        ds_attrs = dict(ds.attrs)
    return {
        "var": var,
        "var_label": str(attrs.get("long_name") or attrs.get("standard_name") or var),
        "unit": str(attrs.get("units") or fallback_unit or ""),
        "prediction_path": str(prediction_path),
        "idownscale_experiment": str(ds_attrs.get("idownscale_experiment", "")),
        "idownscale_test_name": str(ds_attrs.get("idownscale_test_name", "")),
        "idownscale_simu_test": str(ds_attrs.get("idownscale_simu_test", "")),
        "idownscale_sample_dir": str(ds_attrs.get("idownscale_sample_dir", "")),
        "idownscale_perfect_model_input_source": str(ds_attrs.get("idownscale_perfect_model_input_source", "")),
        "idownscale_perfect_model_target_source": str(ds_attrs.get("idownscale_perfect_model_target_source", "")),
        "idownscale_perfect_model_input_resolution": str(ds_attrs.get("idownscale_perfect_model_input_resolution", "")),
        "idownscale_perfect_model_target_resolution": str(ds_attrs.get("idownscale_perfect_model_target_resolution", "")),
    }


@dataclass
class Accumulator:
    count: int = 0
    sum_diff: float = 0.0
    sum_abs_diff: float = 0.0
    sum_sq_diff: float = 0.0
    sum_a: float = 0.0
    sum_b: float = 0.0
    sum_a_sq: float = 0.0
    sum_b_sq: float = 0.0
    sum_ab: float = 0.0

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            return
        av = a[mask].astype(np.float64, copy=False)
        bv = b[mask].astype(np.float64, copy=False)
        diff = av - bv
        self.count += int(diff.size)
        self.sum_diff += float(diff.sum())
        self.sum_abs_diff += float(np.abs(diff).sum())
        self.sum_sq_diff += float((diff**2).sum())
        self.sum_a += float(av.sum())
        self.sum_b += float(bv.sum())
        self.sum_a_sq += float((av**2).sum())
        self.sum_b_sq += float((bv**2).sum())
        self.sum_ab += float((av * bv).sum())

    def row(self, window: str, comparison: str) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {"window": window, "comparison": comparison, "count": self.count}
        if self.count == 0:
            return row
        mean_a = self.sum_a / self.count
        mean_b = self.sum_b / self.count
        var_a = max(self.sum_a_sq / self.count - mean_a**2, 0.0)
        var_b = max(self.sum_b_sq / self.count - mean_b**2, 0.0)
        cov = self.sum_ab / self.count - mean_a * mean_b
        row.update(
            {
                "a_mean": mean_a,
                "truth_mean": mean_b,
                "bias_a_minus_truth": self.sum_diff / self.count,
                "mean_abs_diff": self.sum_abs_diff / self.count,
                "rmse": float(np.sqrt(self.sum_sq_diff / self.count)),
                "a_std": float(np.sqrt(var_a)),
                "truth_std": float(np.sqrt(var_b)),
                "corr": float(cov / np.sqrt(var_a * var_b)) if var_a > 0 and var_b > 0 else np.nan,
            }
        )
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="perfect_model_rcm")
    parser.add_argument("--test-name", default="unet_perfect_model_rcm")
    parser.add_argument("--simu-test", default="rcm")
    parser.add_argument("--var", default="tas", help="Variable name in the prediction NetCDF and sample tensors.")
    parser.add_argument("--startdate", required=True)
    parser.add_argument("--enddate", required=True)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--unit", default=None, help="Fallback unit if the prediction NetCDF does not carry one.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stem-suffix", default="")
    return parser.parse_args()


def _markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [cols, ["---"] * len(cols)]
    for _, record in df.iterrows():
        rendered = []
        for value in record.tolist():
            if isinstance(value, float):
                rendered.append(f"{value:.6f}" if np.isfinite(value) else "nan")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> int:
    args = parse_args()
    sample_dir = Path(args.sample_dir) if args.sample_dir else Path(CONFIG[args.exp]["dataset"])
    test_name = f"{args.test_name}_{args.simu_test}"
    prediction_path = get_prediction_output_path(args.exp, args.simu_test, args.var, args.startdate, args.enddate, test_name)
    output_dir = Path(args.output_dir) if args.output_dir else METRICS_DIR / args.exp / "comparison_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_channel_index, target_channel_index = resolve_channel_indices(args.exp, args.var)
    metadata = get_prediction_metadata(prediction_path, args.var, args.unit)

    ml = Accumulator()
    raw = Accumulator()
    with xr.open_dataset(prediction_path) as ds:
        dates = pd.to_datetime(ds.time.values)
        for i, date in enumerate(dates):
            sample = np.load(sample_dir / f"sample_{pd.Timestamp(date).strftime('%Y%m%d')}.npz")
            truth = sample["y"][target_channel_index]
            raw_input = sample["x"][input_channel_index]
            prediction = ds[args.var].isel(time=i).values
            ml.update(prediction, truth)
            raw.update(raw_input, truth)

    window = f"{args.startdate}_{args.enddate}"
    df = pd.DataFrame([ml.row(window, "ml_minus_truth"), raw.row(window, "raw_input_minus_truth")])
    for key, value in metadata.items():
        df[key] = value
    df["input_channel_index"] = input_channel_index
    df["target_channel_index"] = target_channel_index
    stem = f"perfect_model_predictions_vs_truth_{args.exp}_{test_name}{args.stem_suffix}"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(
        "\n".join(
            [
                "# Perfect-Model Predictions vs Pseudo-Truth",
                "",
                f"- exp: `{args.exp}`",
                f"- test-name: `{args.test_name}`",
                f"- simu-test: `{args.simu_test}`",
                f"- variable: `{metadata['var']}`",
                f"- variable label: `{metadata['var_label']}`",
                f"- unit: `{metadata['unit'] or 'not available'}`",
                f"- prediction: `{prediction_path}`",
                f"- sample-dir: `{sample_dir}`",
                f"- input channel index: `{input_channel_index}`",
                f"- target channel index: `{target_channel_index}`",
                "",
                _markdown(df),
                "",
            ]
        )
    )
    print(f"comparison_csv={csv_path}")
    print(f"comparison_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
