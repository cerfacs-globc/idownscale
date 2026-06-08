#!/usr/bin/env python3
"""
Validate BC outputs scientifically and, optionally, against an archival reference.

The scientific side focuses on simple, fast checks that are easy to interpret:
- mean and standard deviation
- RMSE and mean bias against ERA5 on historical periods
- q05/q50/q95 quantile biases
- lightweight PDF plots

The parity side compares:
- bc_train_hist / bc_test_hist / bc_test_future NPZ volumes
- corrected packaged samples under dataset_<exp>_test_<simu>_bc
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(".")

from iriscc.settings import (
    DATASET_BC_DIR,
    EXP5_ARCHIVE_DATASET_DIR,
    GRAPHS_DIR,
    METRICS_DIR,
    get_bias_corrected_netcdf_path,
)


QUANTILES = (0.05, 0.50, 0.95)


@dataclass
class StreamingDiff:
    count: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    max_abs: float = 0.0

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            return
        diff = (a[mask] - b[mask]).astype(np.float64, copy=False)
        abs_diff = np.abs(diff)
        self.count += int(diff.size)
        self.sum_abs += float(abs_diff.sum())
        self.sum_sq += float((diff**2).sum())
        self.max_abs = max(self.max_abs, float(abs_diff.max()))

    def as_dict(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "count": 0,
                "mean_abs_diff": float("nan"),
                "rmse": float("nan"),
                "max_abs_diff": float("nan"),
            }
        return {
            "count": self.count,
            "mean_abs_diff": self.sum_abs / self.count,
            "rmse": float(np.sqrt(self.sum_sq / self.count)),
            "max_abs_diff": self.max_abs,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scientific BC validation and archive parity checks.")
    parser.add_argument("--exp", default="exp5")
    parser.add_argument("--simu", required=True, help="Simulation alias or source key, e.g. gcm or rcm")
    parser.add_argument("--var", default="tas")
    parser.add_argument("--ssp", default="ssp585")
    parser.add_argument(
        "--archive-root",
        default=os.getenv("IDOWNSCALE_ARCHIVE_ROOT"),
        help=(
            "Archive root containing datasets/dataset_bc for parity checks. "
            "Can also be set with IDOWNSCALE_ARCHIVE_ROOT."
        ),
    )
    parser.add_argument(
        "--skip-archive-parity",
        action="store_true",
        help="Only produce scientific BC validation and skip archive parity checks.",
    )
    return parser.parse_args()


def flatten_valid(*arrays: np.ndarray) -> list[np.ndarray]:
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return [arr[mask].astype(np.float64, copy=False) for arr in arrays]


def quantile_bias(values: np.ndarray, reference: np.ndarray, q: float) -> float:
    return float(np.nanquantile(values, q) - np.nanquantile(reference, q))


def summarize_hist_period(period: str, obs: np.ndarray, raw: np.ndarray, bc: np.ndarray) -> list[dict[str, object]]:
    obs_v, raw_v, bc_v = flatten_valid(obs, raw, bc)
    rows: list[dict[str, object]] = []
    for label, data in (("obs", obs_v), ("raw", raw_v), ("bc", bc_v)):
        rows.append(
            {
                "period": period,
                "series": label,
                "count": int(data.size),
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "q05": float(np.nanquantile(data, 0.05)),
                "q50": float(np.nanquantile(data, 0.50)),
                "q95": float(np.nanquantile(data, 0.95)),
                "mean_bias_vs_obs": 0.0 if label == "obs" else float(np.nanmean(data - obs_v)),
                "rmse_vs_obs": 0.0 if label == "obs" else float(np.sqrt(np.nanmean((data - obs_v) ** 2))),
                "std_delta_vs_obs": 0.0 if label == "obs" else float(np.nanstd(data) - np.nanstd(obs_v)),
                "q05_bias_vs_obs": 0.0 if label == "obs" else quantile_bias(data, obs_v, 0.05),
                "q50_bias_vs_obs": 0.0 if label == "obs" else quantile_bias(data, obs_v, 0.50),
                "q95_bias_vs_obs": 0.0 if label == "obs" else quantile_bias(data, obs_v, 0.95),
            }
        )
    return rows


def summarize_future_period(period: str, hist_raw: np.ndarray, hist_bc: np.ndarray, future_raw: np.ndarray, future_bc: np.ndarray) -> list[dict[str, object]]:
    hist_raw_v, hist_bc_v = flatten_valid(hist_raw, hist_bc)
    future_raw_v = future_raw[np.isfinite(future_raw)].astype(np.float64, copy=False)
    future_bc_v = future_bc[np.isfinite(future_bc)].astype(np.float64, copy=False)
    rows: list[dict[str, object]] = []
    for label, data, hist_ref in (("raw", future_raw_v, hist_raw_v), ("bc", future_bc_v, hist_bc_v)):
        rows.append(
            {
                "period": period,
                "series": label,
                "count": int(data.size),
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "q05": float(np.nanquantile(data, 0.05)),
                "q50": float(np.nanquantile(data, 0.50)),
                "q95": float(np.nanquantile(data, 0.95)),
                "mean_shift_vs_hist_test": float(np.nanmean(data) - np.nanmean(hist_ref)),
                "std_shift_vs_hist_test": float(np.nanstd(data) - np.nanstd(hist_ref)),
                "q05_shift_vs_hist_test": quantile_bias(data, hist_ref, 0.05),
                "q50_shift_vs_hist_test": quantile_bias(data, hist_ref, 0.50),
                "q95_shift_vs_hist_test": quantile_bias(data, hist_ref, 0.95),
            }
        )
    return rows


def load_current_bc(exp: str, simu: str, var: str, ssp: str) -> tuple[dict, dict, dict, xr.Dataset, xr.Dataset, xr.Dataset]:
    base = DATASET_BC_DIR
    train_hist = dict(np.load(base / f"bc_train_hist_{simu}.npz", allow_pickle=True))
    test_hist = dict(np.load(base / f"bc_test_hist_{simu}.npz", allow_pickle=True))
    test_future = dict(np.load(base / f"bc_test_future_{simu}.npz", allow_pickle=True))

    train_hist_bc = xr.open_dataset(get_bias_corrected_netcdf_path(exp, simu, var, "train_hist", ssp=ssp))
    test_hist_bc = xr.open_dataset(get_bias_corrected_netcdf_path(exp, simu, var, "test_hist", ssp=ssp))
    test_future_bc = xr.open_dataset(get_bias_corrected_netcdf_path(exp, simu, var, "test_future", ssp=ssp))
    return train_hist, test_hist, test_future, train_hist_bc, test_hist_bc, test_future_bc


def align_bundle_with_bc(bundle: dict, ds_bc: xr.Dataset, var: str, simu: str) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    bundle_dates = pd.to_datetime(bundle["dates"])
    bc_dates = pd.to_datetime(ds_bc.time.values)
    common_dates = pd.Index(bundle_dates).intersection(pd.Index(bc_dates))
    if len(common_dates) == 0:
        raise ValueError(f"No overlapping dates between BC bundle and corrected dataset for {simu}")

    bundle_index = pd.Index(bundle_dates).get_indexer(common_dates)
    bc_index = pd.Index(bc_dates).get_indexer(common_dates)

    obs = None
    if "era5" in bundle:
        obs = bundle["era5"][bundle_index]
    raw = bundle[simu][bundle_index]
    bc = ds_bc[var].isel(time=bc_index).values
    return obs, raw, bc


def make_pdf_plot(
    output_path: Path,
    train_hist_obs: np.ndarray,
    train_hist_raw: np.ndarray,
    train_hist_bc: np.ndarray,
    test_hist_obs: np.ndarray,
    test_hist_raw: np.ndarray,
    test_hist_bc: np.ndarray,
    test_future_raw: np.ndarray,
    test_future_bc: np.ndarray,
    simu: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [
        ("1980-1999 train", train_hist_obs, train_hist_raw, train_hist_bc),
        ("2000-2014 test", test_hist_obs, test_hist_raw, test_hist_bc),
        ("2015-2100 future", None, test_future_raw, test_future_bc),
    ]
    for ax, (title, obs, raw, bc) in zip(axes, panels):
        series = []
        if obs is not None:
            series.append(("ERA5", obs, "black"))
        series.append(("Raw", raw, "gray"))
        series.append(("BC", bc, "blue"))
        for label, data, color in series:
            values = data[np.isfinite(data)].astype(np.float64, copy=False)
            ax.hist(values, bins=80, density=True, histtype="step", linewidth=1.8, label=label, color=color)
            ax.axvline(float(np.nanmean(values)), color=color, linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Temperature [K]")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].set_ylabel("Density")
    axes[-1].legend(loc="upper left")
    fig.suptitle(f"BC Scientific Validation: {simu}", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compare_npz_volume(candidate_path: Path, archive_path: Path, simu: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidate = dict(np.load(candidate_path, allow_pickle=True))
    archive = dict(np.load(archive_path, allow_pickle=True))
    for key in sorted(set(candidate) | set(archive)):
        row: dict[str, object] = {"artifact": candidate_path.name, "key": key}
        if key not in candidate or key not in archive:
            row["status"] = "missing_key"
            row["candidate_present"] = key in candidate
            row["archive_present"] = key in archive
            rows.append(row)
            continue
        cand = np.asarray(candidate[key])
        ref = np.asarray(archive[key])
        row["candidate_shape"] = tuple(cand.shape)
        row["archive_shape"] = tuple(ref.shape)
        if cand.shape != ref.shape:
            row["status"] = "shape_mismatch"
            rows.append(row)
            continue
        if cand.dtype.kind in {"M", "U", "S", "O"}:
            row["status"] = "ok"
            row["exact_match"] = bool(np.array_equal(cand, ref))
            rows.append(row)
            continue
        diff = cand.astype(np.float64) - ref.astype(np.float64)
        abs_diff = np.abs(diff[np.isfinite(diff)])
        row["status"] = "ok"
        row["mean_abs_diff"] = float(abs_diff.mean()) if abs_diff.size else float("nan")
        row["rmse"] = float(np.sqrt(np.mean(diff[np.isfinite(diff)] ** 2))) if abs_diff.size else float("nan")
        row["max_abs_diff"] = float(abs_diff.max()) if abs_diff.size else float("nan")
        row["candidate_mean"] = float(np.nanmean(cand))
        row["archive_mean"] = float(np.nanmean(ref))
        row["candidate_std"] = float(np.nanstd(cand))
        row["archive_std"] = float(np.nanstd(ref))
        if key == simu and "era5" in candidate and "era5" in archive and candidate["era5"].shape == archive["era5"].shape:
            row["candidate_mean_bias_vs_era5"] = float(np.nanmean(cand - candidate["era5"]))
            row["archive_mean_bias_vs_era5"] = float(np.nanmean(ref - archive["era5"]))
        rows.append(row)
    return rows


def iter_samples(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("sample_*.npz"))


def compare_corrected_samples(candidate_dir: Path, archive_dir: Path) -> list[dict[str, object]]:
    candidate_files = {path.name: path for path in iter_samples(candidate_dir)}
    archive_files = {path.name: path for path in iter_samples(archive_dir)}
    common = sorted(set(candidate_files) & set(archive_files))

    x_stats = StreamingDiff()
    y_stats = StreamingDiff()
    missing_candidate = sorted(set(archive_files) - set(candidate_files))
    missing_archive = sorted(set(candidate_files) - set(archive_files))
    key_mismatches = 0
    shape_mismatches = 0

    for name in common:
        cand_npz = np.load(candidate_files[name])
        ref_npz = np.load(archive_files[name])
        cand_keys = set(cand_npz.files)
        ref_keys = set(ref_npz.files)
        if cand_keys != ref_keys:
            key_mismatches += 1
            continue
        if cand_npz["x"].shape != ref_npz["x"].shape:
            shape_mismatches += 1
            continue
        x_stats.update(cand_npz["x"], ref_npz["x"])
        if "y" in cand_keys:
            if cand_npz["y"].shape != ref_npz["y"].shape:
                shape_mismatches += 1
                continue
            y_stats.update(cand_npz["y"], ref_npz["y"])

    return [
        {"artifact": "corrected_samples", "metric": "candidate_count", "value": len(candidate_files)},
        {"artifact": "corrected_samples", "metric": "archive_count", "value": len(archive_files)},
        {"artifact": "corrected_samples", "metric": "common_count", "value": len(common)},
        {"artifact": "corrected_samples", "metric": "missing_in_candidate", "value": len(missing_candidate)},
        {"artifact": "corrected_samples", "metric": "missing_in_archive", "value": len(missing_archive)},
        {"artifact": "corrected_samples", "metric": "key_mismatches", "value": key_mismatches},
        {"artifact": "corrected_samples", "metric": "shape_mismatches", "value": shape_mismatches},
        {"artifact": "corrected_samples_x", **x_stats.as_dict()},
        {"artifact": "corrected_samples_y", **y_stats.as_dict()},
    ]


def main() -> int:
    args = parse_args()

    train_hist, test_hist, test_future, train_hist_bc_ds, test_hist_bc_ds, test_future_bc_ds = load_current_bc(
        args.exp,
        args.simu,
        args.var,
        args.ssp,
    )

    train_hist_obs, train_hist_raw, train_hist_bc = align_bundle_with_bc(train_hist, train_hist_bc_ds, args.var, args.simu)
    test_hist_obs, test_hist_raw, test_hist_bc = align_bundle_with_bc(test_hist, test_hist_bc_ds, args.var, args.simu)
    _, test_future_raw, test_future_bc = align_bundle_with_bc(test_future, test_future_bc_ds, args.var, args.simu)

    summary_rows: list[dict[str, object]] = []
    summary_rows += summarize_hist_period("train_hist", train_hist_obs, train_hist_raw, train_hist_bc)
    summary_rows += summarize_hist_period("test_hist", test_hist_obs, test_hist_raw, test_hist_bc)
    summary_rows += summarize_future_period("test_future", test_hist_raw, test_hist_bc, test_future_raw, test_future_bc)
    summary_df = pd.DataFrame(summary_rows)

    metrics_dir = METRICS_DIR / args.exp
    graphs_dir = GRAPHS_DIR / "metrics" / args.exp
    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = metrics_dir / f"bc_validation_summary_{args.exp}_{args.simu}.csv"
    summary_df.to_csv(summary_csv, index=False)

    pdf_png = graphs_dir / f"bc_validation_pdf_{args.exp}_{args.simu}.png"
    make_pdf_plot(
        pdf_png,
        train_hist_obs,
        train_hist_raw,
        train_hist_bc,
        test_hist_obs,
        test_hist_raw,
        test_hist_bc,
        test_future_raw,
        test_future_bc,
        args.simu,
    )

    print(f"scientific_summary={summary_csv}")
    print(f"scientific_plot={pdf_png}")

    if args.skip_archive_parity:
        return 0

    if not args.archive_root:
        raise SystemExit(
            "Archive parity requested but no archive root was provided. "
            "Pass --archive-root, set IDOWNSCALE_ARCHIVE_ROOT, or use --skip-archive-parity."
        )

    archive_dataset_bc_dir = Path(args.archive_root) / "datasets" / "dataset_bc"
    parity_rows: list[dict[str, object]] = []
    for period in ("train_hist", "test_hist", "test_future"):
        parity_rows += compare_npz_volume(
            DATASET_BC_DIR / f"bc_{period}_{args.simu}.npz",
            archive_dataset_bc_dir / f"bc_{period}_{args.simu}.npz",
            args.simu,
        )

    parity_df = pd.DataFrame(parity_rows)
    parity_csv = metrics_dir / f"bc_archive_parity_{args.exp}_{args.simu}.csv"
    parity_df.to_csv(parity_csv, index=False)
    print(f"archive_parity={parity_csv}")

    candidate_corrected_dir = DATASET_BC_DIR / f"dataset_{args.exp}_test_{args.simu}_bc"
    archive_corrected_dir = archive_dataset_bc_dir / f"dataset_{args.exp}_test_{args.simu}_bc"
    if candidate_corrected_dir.exists() and archive_corrected_dir.exists():
        sample_parity_df = pd.DataFrame(compare_corrected_samples(candidate_corrected_dir, archive_corrected_dir))
        sample_parity_csv = metrics_dir / f"bc_archive_corrected_sample_parity_{args.exp}_{args.simu}.csv"
        sample_parity_df.to_csv(sample_parity_csv, index=False)
        print(f"archive_corrected_sample_parity={sample_parity_csv}")
    else:
        print(
            "archive_corrected_sample_parity=skipped "
            f"candidate_exists={candidate_corrected_dir.exists()} archive_exists={archive_corrected_dir.exists()}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
