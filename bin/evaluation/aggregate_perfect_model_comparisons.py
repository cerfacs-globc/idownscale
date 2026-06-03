#!/usr/bin/env python3
"""Aggregate perfect-model comparison chunks into model summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(".")

from iriscc.settings import METRICS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="perfect_model_rcm")
    parser.add_argument("--simu-test", default="rcm")
    parser.add_argument("--var", default="tas")
    parser.add_argument("--test-name", action="append", required=True, help="Model run name. Repeat for many models.")
    parser.add_argument("--chunks-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def format_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}" if np.isfinite(value) else "nan"
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    rows = [list(df.columns), ["---"] * len(df.columns)]
    rows.extend([[format_value(value) for value in record.tolist()] for _, record in df.iterrows()])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def aggregate_model(chunks_dir: Path, exp: str, test_name: str, simu_test: str, var: str) -> pd.DataFrame:
    pattern = f"perfect_model_predictions_vs_truth_{exp}_{test_name}_{simu_test}_*.csv"
    rows: list[dict[str, float | str]] = []
    for path in sorted(chunks_dir.glob(pattern)):
        df = pd.read_csv(path)
        ml_rows = df[df["comparison"] == "ml_minus_truth"]
        raw_rows = df[df["comparison"] == "raw_input_minus_truth"]
        if ml_rows.empty or raw_rows.empty:
            raise ValueError(f"Missing comparison rows in {path}")
        ml = ml_rows.iloc[0]
        raw = raw_rows.iloc[0]
        rows.append(
            {
                "window": ml["window"],
                "var": ml.get("var", var),
                "var_label": ml.get("var_label", ""),
                "unit": ml.get("unit", ""),
                "ml_bias": ml["bias_a_minus_truth"],
                "ml_rmse": ml["rmse"],
                "raw_bias": raw["bias_a_minus_truth"],
                "raw_rmse": raw["rmse"],
                "rmse_reduction": raw["rmse"] - ml["rmse"],
                "ml_corr": ml["corr"],
                "raw_corr": raw["corr"],
                "ml_std": ml["a_std"],
                "truth_std": ml["truth_std"],
                "input_channel_index": ml.get("input_channel_index", np.nan),
                "target_channel_index": ml.get("target_channel_index", np.nan),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No chunk CSVs matched {chunks_dir / pattern}")
    df = pd.DataFrame(rows)
    for col in ("var", "var_label", "unit"):
        if col in df.columns:
            cleaned = df[col].replace("", np.nan)
            if cleaned.notna().any():
                df[col] = cleaned.ffill().bfill().fillna("")
    # Keep legacy temperature-shaped aliases for existing notebooks and tables.
    df["ml_bias_K"] = df["ml_bias"]
    df["ml_rmse_K"] = df["ml_rmse"]
    df["raw_bias_K"] = df["raw_bias"]
    df["raw_rmse_K"] = df["raw_rmse"]
    df["rmse_gain_K"] = df["rmse_reduction"]
    df["ml_std_K"] = df["ml_std"]
    df["truth_std_K"] = df["truth_std"]
    return df


def write_model_outputs(output_dir: Path, exp: str, test_name: str, simu_test: str, df: pd.DataFrame) -> None:
    stem = f"perfect_model_predictions_vs_truth_{exp}_{test_name}_{simu_test}"
    var = df["var"].dropna().iloc[0] if "var" in df and df["var"].notna().any() else ""
    var_label = df["var_label"].dropna().iloc[0] if "var_label" in df and df["var_label"].notna().any() else var
    unit = df["unit"].dropna().iloc[0] if "unit" in df and df["unit"].notna().any() else ""
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(
        "\n".join(
            [
                "# Perfect-Model Aggregate Comparison",
                "",
                f"- exp: `{exp}`",
                f"- model: `{test_name}`",
                f"- simu-test: `{simu_test}`",
                f"- variable: `{var}`",
                f"- variable label: `{var_label}`",
                f"- unit: `{unit or 'not available'}`",
                "- comparison: prediction and degraded/coarse input against native RCM pseudo-truth `y`",
                "",
                markdown_table(df),
                "",
            ]
        )
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else METRICS_DIR / args.exp / "comparison_tables"
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    for test_name in args.test_name:
        df = aggregate_model(chunks_dir, args.exp, test_name, args.simu_test, args.var)
        write_model_outputs(output_dir, args.exp, test_name, args.simu_test, df)
        model_df = df.copy()
        model_df.insert(0, "model", test_name)
        combined.append(model_df)

    all_df = pd.concat(combined, ignore_index=True)
    combined_stem = f"perfect_model_predictions_vs_truth_{args.exp}_combined_{args.simu_test}"
    all_df.to_csv(output_dir / f"{combined_stem}.csv", index=False)
    var = all_df["var"].dropna().iloc[0] if "var" in all_df and all_df["var"].notna().any() else ""
    var_label = all_df["var_label"].dropna().iloc[0] if "var_label" in all_df and all_df["var_label"].notna().any() else var
    unit = all_df["unit"].dropna().iloc[0] if "unit" in all_df and all_df["unit"].notna().any() else ""
    display_cols = [
        "model",
        "window",
        "var",
        "unit",
        "ml_bias",
        "ml_rmse",
        "raw_bias",
        "raw_rmse",
        "rmse_reduction",
        "ml_corr",
        "raw_corr",
    ]
    (output_dir / f"{combined_stem}.md").write_text(
        "\n".join(
            [
                "# Perfect-Model Combined Comparison",
                "",
                f"- variable: `{var}`",
                f"- variable label: `{var_label}`",
                f"- unit: `{unit or 'not available'}`",
                "- comparison: ML prediction and degraded/coarse input against native RCM pseudo-truth `y`",
                "- lower absolute bias/RMSE is better; `rmse_reduction = raw_rmse - ml_rmse`",
                "",
                markdown_table(all_df[display_cols]),
                "",
            ]
        )
    )
    print(f"combined_csv={output_dir / f'{combined_stem}.csv'}")
    print(f"combined_md={output_dir / f'{combined_stem}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
