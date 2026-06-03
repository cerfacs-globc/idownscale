#!/usr/bin/env python3
"""Validate perfect-model sample datasets before downstream prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(".")

from iriscc.settings import CONFIG, DATASET_BC_DIR, METRICS_DIR


QUANTILES = [0, 1, 5, 50, 95, 99, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate perfect-model raw sample structure and value distributions.")
    parser.add_argument("--exp", default="exp5")
    parser.add_argument("--simu", default="rcm")
    parser.add_argument("--var", default="tas", help="Variable name used only for labels in validation outputs.")
    parser.add_argument("--unit", default="", help="Optional unit label for the validation markdown and CSV outputs.")
    parser.add_argument("--startdate", default="20000101")
    parser.add_argument("--enddate", default="21001231")
    parser.add_argument("--historical-enddate", default="20141231")
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument(
        "--allow-extra-samples",
        action="store_true",
        help="Allow the sample directory to contain dates outside the validated window.",
    )
    parser.add_argument(
        "--expect-future-y",
        action="store_true",
        help="Require y targets for future dates as well as historical dates.",
    )
    parser.add_argument(
        "--min-cross-period-x-diff",
        type=float,
        default=0.01,
        help="Fail if the selected predictor channel mean absolute difference across same-calendar-day periods is below this value.",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def resolve_channel_indices(exp: str, var: str) -> tuple[int, int]:
    input_vars = CONFIG[exp].get("input_vars", [])
    target_vars = CONFIG[exp].get("target_vars", [])
    input_index = input_vars.index(var) if var in input_vars else max(len(input_vars) - 1, 0)
    target_index = target_vars.index(var) if var in target_vars else 0
    return input_index, target_index


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, **{f"q{q}": np.nan for q in QUANTILES}}
    quantiles = np.nanpercentile(values, QUANTILES)
    row: dict[str, float | int] = {
        "count": int(values.size),
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
    }
    row.update({f"q{q}": float(value) for q, value in zip(QUANTILES, quantiles)})
    return row


def sample_path(sample_dir: Path, date: pd.Timestamp) -> Path:
    return sample_dir / f"sample_{date.strftime('%Y%m%d')}.npz"


def period_windows(start: str, end: str, historical_end: str) -> dict[str, tuple[str, str]]:
    hist_end = pd.Timestamp(historical_end)
    windows = {"hist": (start, historical_end)}
    future_edges = [
        ("future_2015_2029", "20150101", "20291231"),
        ("future_2030_2044", "20300101", "20441231"),
        ("future_2045_2059", "20450101", "20591231"),
        ("future_2060_2074", "20600101", "20741231"),
        ("future_2075_2089", "20750101", "20891231"),
        ("future_2090_2100", "20900101", end),
    ]
    for name, window_start, window_end in future_edges:
        if pd.Timestamp(window_start) <= pd.Timestamp(end) and pd.Timestamp(window_end) > hist_end:
            windows[name] = (window_start, min(pd.Timestamp(window_end), pd.Timestamp(end)).strftime("%Y%m%d"))
    return windows


def sampled_dates(start: str, end: str) -> list[pd.Timestamp]:
    dates = list(pd.date_range(start, end, freq="365D"))
    for extra in ("0115", "0715", "1231"):
        for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            day = pd.Timestamp(f"{year}{extra}")
            if pd.Timestamp(start) <= day <= pd.Timestamp(end):
                dates.append(day)
    return sorted(set(dates))


def collect_values(sample_dir: Path, dates: list[pd.Timestamp], input_index: int, target_index: int) -> tuple[np.ndarray, np.ndarray | None, list[pd.Timestamp]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    used: list[pd.Timestamp] = []
    for date in dates:
        path = sample_path(sample_dir, date)
        if not path.exists():
            continue
        data = np.load(path)
        x_var = data["x"][input_index]
        xs.append(x_var[np.isfinite(x_var)])
        if "y" in data:
            y_var = data["y"][target_index]
            ys.append(y_var[np.isfinite(y_var)])
        used.append(date)
    x = np.concatenate(xs) if xs else np.array([], dtype=np.float32)
    y = np.concatenate(ys) if ys else None
    return x, y, used


def validate_inventory(sample_dir: Path, dates: pd.DatetimeIndex, allow_extra_samples: bool) -> dict[str, object]:
    expected = {f"sample_{date.strftime('%Y%m%d')}.npz" for date in dates}
    actual = {path.name for path in sample_dir.glob("sample_*.npz")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_head": ",".join(missing[:5]),
        "extra_head": ",".join(extra[:5]),
        "status": "ok" if not missing and (allow_extra_samples or not extra) else "fail",
    }


def validate_structure(
    sample_dir: Path,
    dates: list[pd.Timestamp],
    historical_end: pd.Timestamp,
    expect_future_y: bool,
) -> list[dict[str, object]]:
    rows = []
    for date in dates:
        path = sample_path(sample_dir, date)
        if not path.exists():
            rows.append(
                {
                    "date": date.strftime("%Y%m%d"),
                    "keys": "missing",
                    "expected_keys": "n/a",
                    "status": "fail",
                }
            )
            continue
        data = np.load(path)
        files = sorted(data.files)
        expected_keys = ["x", "y"] if expect_future_y or date <= historical_end else ["x"]
        row = {
            "date": date.strftime("%Y%m%d"),
            "keys": ",".join(files),
            "expected_keys": ",".join(expected_keys),
            "x_shape": tuple(data["x"].shape),
            "x_dtype": str(data["x"].dtype),
            "x_finite_fraction": float(np.isfinite(data["x"]).mean()),
            "status": "ok",
        }
        if files != expected_keys:
            row["status"] = "fail"
        if "y" in data:
            row["y_shape"] = tuple(data["y"].shape)
            row["y_dtype"] = str(data["y"].dtype)
            row["y_finite_fraction"] = float(np.isfinite(data["y"]).mean())
        rows.append(row)
    return rows


def validate_cross_period_repeat(
    sample_dir: Path,
    min_x_diff: float,
    input_index: int,
    target_index: int,
    startdate: pd.Timestamp,
    enddate: pd.Timestamp,
) -> pd.DataFrame:
    pairs = [
        ("jan01_hist_future", "20000101", "20150101"),
        ("jan01_future_mid", "20150101", "20300101"),
        ("dec31_hist_future", "20141231", "20291231"),
        ("dec31_future_late", "20291231", "21001231"),
    ]
    rows = []
    for label, left_date, right_date in pairs:
        left_path = sample_dir / f"sample_{left_date}.npz"
        right_path = sample_dir / f"sample_{right_date}.npz"
        row = {
            "pair": label,
            "left_date": left_date,
            "right_date": right_date,
            "status": "missing",
        }
        if pd.Timestamp(left_date) < startdate or pd.Timestamp(right_date) > enddate:
            row["status"] = "skipped"
            rows.append(row)
            continue
        if left_path.exists() and right_path.exists():
            left = np.load(left_path)
            right = np.load(right_path)
            x_diff = float(np.nanmean(np.abs(left["x"][input_index] - right["x"][input_index])))
            row.update(
                {
                    "x_absdiff_mean": x_diff,
                    "status": "ok" if x_diff >= min_x_diff else "fail",
                }
            )
            if "y" in left and "y" in right:
                row["y_absdiff_mean"] = float(np.nanmean(np.abs(left["y"][target_index] - right["y"][target_index])))
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(
    path: Path,
    metadata: dict[str, object],
    inventory: dict[str, object],
    structure: pd.DataFrame,
    repeat_checks: pd.DataFrame,
    period_stats: pd.DataFrame,
    edge_stats: pd.DataFrame,
) -> None:
    lines = [
        "# Perfect-Model Sample Validation",
        "",
        "## Metadata",
        "",
        inventory_to_markdown(metadata),
        "",
        "## Inventory",
        "",
        inventory_to_markdown(inventory),
        "",
        "## Structure Checks",
        "",
        structure.to_markdown(index=False),
        "",
        "## Cross-Period Predictor Repeat Checks",
        "",
        repeat_checks.to_markdown(index=False),
        "",
        "## Period Value Distributions",
        "",
        period_stats.to_markdown(index=False),
        "",
        "## Boundary-Day Value Distributions",
        "",
        edge_stats.to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines))


def inventory_to_markdown(inventory: dict[str, object]) -> str:
    return "\n".join(f"- `{key}`: `{value}`" for key, value in inventory.items())


def main() -> int:
    args = parse_args()
    sample_dir = Path(args.sample_dir) if args.sample_dir else DATASET_BC_DIR / f"dataset_{args.exp}_test_{args.simu}"
    output_dir = Path(args.output_dir) if args.output_dir else METRICS_DIR / args.exp / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_channel_index, target_channel_index = resolve_channel_indices(args.exp, args.var)

    dates = pd.date_range(args.startdate, args.enddate, freq="D")
    historical_end = pd.Timestamp(args.historical_enddate)
    inventory = validate_inventory(sample_dir, dates, args.allow_extra_samples)

    structure_dates = [
        pd.Timestamp(args.startdate),
        historical_end,
        historical_end + pd.Timedelta(days=1),
        pd.Timestamp(args.enddate),
    ]
    if pd.Timestamp("20700101") <= pd.Timestamp(args.enddate):
        structure_dates.insert(-1, pd.Timestamp("20700101"))
    structure_dates = [date for date in structure_dates if pd.Timestamp(args.startdate) <= date <= pd.Timestamp(args.enddate)]
    structure_dates = sorted(set(structure_dates))
    structure_rows = validate_structure(sample_dir, structure_dates, historical_end, args.expect_future_y)
    structure_df = pd.DataFrame(structure_rows)
    repeat_df = validate_cross_period_repeat(
        sample_dir,
        args.min_cross_period_x_diff,
        input_channel_index,
        target_channel_index,
        pd.Timestamp(args.startdate),
        pd.Timestamp(args.enddate),
    )
    metadata = {
        "exp": args.exp,
        "simu": args.simu,
        "var": args.var,
        "unit": args.unit or "",
        "sample_dir": sample_dir,
        "input_channel_index": input_channel_index,
        "target_channel_index": target_channel_index,
    }

    period_rows = []
    for period, (start, end) in period_windows(args.startdate, args.enddate, args.historical_enddate).items():
        x, y, used = collect_values(sample_dir, sampled_dates(start, end), input_channel_index, target_channel_index)
        row = {"period": period, "series": f"x_{args.var}", "sampled_days": len(used)}
        row.update(stats(x))
        row["unit"] = args.unit or ""
        period_rows.append(row)
        if y is not None:
            row = {"period": period, "series": f"y_{args.var}", "sampled_days": len(used)}
            row.update(stats(y))
            row["unit"] = args.unit or ""
            period_rows.append(row)
    period_df = pd.DataFrame(period_rows)

    edge_rows = []
    edge_dates = set(pd.date_range(historical_end - pd.Timedelta(days=2), historical_end + pd.Timedelta(days=3), freq="D"))
    edge_dates.update(pd.date_range(pd.Timestamp(args.enddate) - pd.Timedelta(days=2), args.enddate, freq="D"))
    edge_dates = sorted(
        date for date in edge_dates if pd.Timestamp(args.startdate) <= date <= pd.Timestamp(args.enddate)
    )
    for date in edge_dates:
        path = sample_path(sample_dir, date)
        if not path.exists():
            edge_rows.append({"date": date.strftime("%Y%m%d"), "keys": "missing", "series": f"x_{args.var}", "unit": args.unit or "", "count": 0, "mean": np.nan, "std": np.nan, **{f'q{q}': np.nan for q in QUANTILES}})
            continue
        data = np.load(path)
        row = {"date": date.strftime("%Y%m%d"), "keys": ",".join(sorted(data.files)), "series": f"x_{args.var}"}
        row.update(stats(data["x"][input_channel_index]))
        row["unit"] = args.unit or ""
        edge_rows.append(row)
        if "y" in data:
            row = {"date": date.strftime("%Y%m%d"), "keys": ",".join(sorted(data.files)), "series": f"y_{args.var}"}
            row.update(stats(data["y"][target_channel_index]))
            row["unit"] = args.unit or ""
            edge_rows.append(row)
    edge_df = pd.DataFrame(edge_rows)

    stem = f"perfect_model_samples_{args.exp}_{args.simu}_{args.startdate}_{args.enddate}"
    pd.DataFrame([inventory]).to_csv(output_dir / f"{stem}_inventory.csv", index=False)
    structure_df.to_csv(output_dir / f"{stem}_structure.csv", index=False)
    repeat_df.to_csv(output_dir / f"{stem}_cross_period_repeat.csv", index=False)
    period_df.to_csv(output_dir / f"{stem}_period_stats.csv", index=False)
    edge_df.to_csv(output_dir / f"{stem}_edge_stats.csv", index=False)
    write_markdown(output_dir / f"{stem}.md", metadata, inventory, structure_df, repeat_df, period_df, edge_df)

    print(f"validation_md={output_dir / f'{stem}.md'}")
    print(f"inventory_status={inventory['status']}")
    print(f"structure_status={'ok' if (structure_df['status'] == 'ok').all() else 'fail'}")
    repeat_status = "ok" if repeat_df["status"].isin(["ok", "skipped"]).all() else "fail"
    print(f"repeat_status={repeat_status}")
    return 0 if inventory["status"] == "ok" and (structure_df["status"] == "ok").all() and repeat_status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
