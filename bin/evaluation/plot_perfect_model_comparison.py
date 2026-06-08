#!/usr/bin/env python3
"""Plot perfect-model ML-vs-pseudo-truth comparison summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(".")

from iriscc.settings import GRAPHS_DIR, METRICS_DIR


MODEL_LABELS = {
    "bc_baseline": "BC baseline",
    "bc_baseline_sbck_cdft": "SBCK CDFt baseline",
    "unet_outputnorm_perfect_model_rcm": "UNet + output norm",
    "unet_perfect_model_rcm": "UNet",
    "unet_rep3_perfect_model_rcm": "UNet replicate",
    "miniunet_perfect_model_rcm": "MiniUNet",
    "unet_seed2_perfect_model_rcm": "UNet cold replicate",
}

MODEL_COLORS = {
    "bc_baseline": "#2A9D8F",
    "bc_baseline_sbck_cdft": "#5B8E7D",
    "unet_outputnorm_perfect_model_rcm": "#006D77",
    "unet_perfect_model_rcm": "#E76F51",
    "unet_rep3_perfect_model_rcm": "#457B9D",
    "miniunet_perfect_model_rcm": "#8D6E63",
    "unet_seed2_perfect_model_rcm": "#B56576",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", default="perfect_model_rcm")
    parser.add_argument("--simu-test", default="rcm")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--var", default="tas")
    parser.add_argument("--unit", default="K", help="Fallback unit when the comparison table has temperature-style columns.")
    parser.add_argument("--bias-tolerance", type=float, default=None, help="Optional absolute bias tolerance to visualize.")
    return parser.parse_args()


def window_label(window: str) -> str:
    start, end = window.split("_")
    return f"{start[:4]}-{end[:4]}"


def model_label(model: str) -> str:
    if model.startswith("bc_baseline_") and model not in MODEL_LABELS:
        suffix = model[len("bc_baseline_") :].replace("_", " ")
        return f"BC baseline ({suffix})"
    return MODEL_LABELS.get(model, model)


def model_color(model: str) -> str:
    if model.startswith("bc_baseline_") and model not in MODEL_COLORS:
        return "#3D9970"
    return MODEL_COLORS.get(model, "#4A4A4A")


def grouped_window_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    model_order: list[str],
    window_order: list[str],
    value_col: str,
    title: str,
    ylabel: str,
    zero_line: bool = False,
) -> None:
    width = 0.8 / max(len(window_order), 1)
    x = np.arange(len(model_order))
    offsets = (np.arange(len(window_order)) - (len(window_order) - 1) / 2) * width
    indexed = df.set_index(["model", "window_label"])
    for idx, window in enumerate(window_order):
        values = [indexed.loc[(model, window), value_col] for model in model_order]
        ax.bar(x + offsets[idx], values, width=width, label=window, color=f"C{idx}", alpha=0.82)
    if zero_line:
        ax.axhline(0, color="#222222", lw=1.1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, [model_label(model) for model in model_order], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)


def main() -> int:
    args = parse_args()
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

    df = pd.read_csv(input_csv)
    if "unit" in df and df["unit"].notna().any():
        unit = str(df["unit"].dropna().iloc[0]) or args.unit
    else:
        unit = args.unit
    if "var_label" in df and df["var_label"].notna().any():
        var_label = str(df["var_label"].dropna().iloc[0])
    else:
        var_label = args.var
    ml_rmse_col = "ml_rmse" if "ml_rmse" in df.columns else "ml_rmse_K"
    raw_rmse_col = "raw_rmse" if "raw_rmse" in df.columns else "raw_rmse_K"
    ml_bias_col = "ml_bias" if "ml_bias" in df.columns else "ml_bias_K"
    rmse_reduction_col = "rmse_reduction" if "rmse_reduction" in df.columns else "rmse_gain_K"
    unit_suffix = f" [{unit}]" if unit else ""
    df["window_label"] = df["window"].map(window_label)
    model_order = [model for model in MODEL_LABELS if model in set(df["model"])]
    model_order += [model for model in df["model"].unique() if model not in MODEL_LABELS]
    window_order = list(dict.fromkeys(df["window_label"]))

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    fig.suptitle(
        f"Perfect-model downscaling: {var_label} predictions vs native RCM pseudo-truth",
        fontsize=16,
        fontweight="bold",
    )

    grouped_window_bars(
        axes[0, 0],
        df,
        model_order,
        window_order,
        ml_rmse_col,
        "RMSE against pseudo-truth",
        f"RMSE{unit_suffix}",
    )
    grouped_window_bars(
        axes[0, 1],
        df,
        model_order,
        window_order,
        ml_bias_col,
        "Mean bias against pseudo-truth",
        f"Bias{unit_suffix}",
        zero_line=True,
    )
    if args.bias_tolerance is not None:
        axes[0, 1].axhspan(
            -args.bias_tolerance,
            args.bias_tolerance,
            color="#2A9D8F",
            alpha=0.12,
            label=f"+/-{args.bias_tolerance:.2f}{unit_suffix} tolerance",
        )
    grouped_window_bars(
        axes[1, 0],
        df,
        model_order,
        window_order,
        rmse_reduction_col,
        "RMSE improvement over coarse-resolution input",
        f"Coarse input RMSE - method RMSE{unit_suffix}",
        zero_line=True,
    )

    ax = axes[1, 1]
    summary = (
        df.groupby("model", as_index=False)
        .agg(mean_rmse=("ml_rmse", "mean") if "ml_rmse" in df.columns else ("ml_rmse_K", "mean"),
             max_abs_bias=("ml_bias", lambda s: float(np.abs(s).max())) if "ml_bias" in df.columns else ("ml_bias_K", lambda s: float(np.abs(s).max())))
        .set_index("model")
        .loc[model_order]
        .reset_index()
    )
    y = np.arange(len(summary))
    rmse_bars = ax.barh(
        y - 0.18,
        summary["mean_rmse"],
        height=0.34,
        color=[model_color(m) for m in summary["model"]],
        label=f"Mean RMSE [{unit}]",
    )
    bias_bars = ax.barh(
        y + 0.18,
        summary["max_abs_bias"],
        height=0.34,
        color="#F4A261",
        label=f"Max |bias| [{unit}]",
    )
    if args.bias_tolerance is not None:
        ax.axvline(args.bias_tolerance, color="#2A9D8F", lw=1.3, ls=":", label=f"{args.bias_tolerance:.2f}{unit_suffix} bias target")
    ax.set_yticks(y, [model_label(m) for m in summary["model"]])
    ax.invert_yaxis()
    ax.set_title("Average RMSE and worst bias")
    ax.set_xlabel(unit or "value")
    ax.grid(axis="x", alpha=0.25)
    ax.bar_label(rmse_bars, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bias_bars, fmt="%.2f", padding=3, fontsize=8)

    handles, labels = [], []
    for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)

    png = output_dir / f"perfect_model_method_comparison_{args.exp}_{args.simu_test}.png"
    pdf = output_dir / f"perfect_model_method_comparison_{args.exp}_{args.simu_test}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"png={png}")
    print(f"pdf={pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
