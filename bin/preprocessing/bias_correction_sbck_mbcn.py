"""
Bias-correct paired variables jointly with SBCK MBCn.

This script is intentionally limited to paired multivariate correction:
- consumes experiment-specific paired BC bundles created with --paired-vars
- writes one canonical corrected NetCDF per variable with a tagged family
- does not materialize corrected samples for ML workflows yet
"""

from __future__ import annotations

import sys

sys.path.append(".")

import argparse

import numpy as np
import xarray as xr

from iriscc.settings import (
    CONFIG,
    get_bc_bundle_path,
    get_bc_train_hist_dates,
    get_bias_corrected_netcdf_path,
    get_simu_family,
    get_simu_source,
    normalize_bc_tag,
)
from iriscc.datautils import Data


DEFAULT_BC_TAG = "sbck_mbcn"


def bundle_obs(bundle: dict) -> np.ndarray:
    if "obs" in bundle:
        return bundle["obs"]
    if "era5" in bundle:
        return bundle["era5"]
    raise KeyError("BC bundle does not contain an 'obs' or legacy 'era5' reference field.")


def corrected_geometry_reference(exp: str, simu: str, simu_source: str) -> str:
    if get_simu_family(exp, simu) == "rcm":
        return CONFIG[exp].get("gcm_source", "gcm_cnrm_cm6_1")
    return simu_source


def build_corrected_dataset(reference_ds: xr.Dataset, var: str, values: np.ndarray, dates: np.ndarray) -> xr.Dataset:
    spatial_dims = reference_ds[var].dims
    coords = {"time": ("time", dates)}
    for coord_name in ("lon", "lat", "x", "y"):
        if coord_name not in reference_ds.coords:
            continue
        coord = reference_ds.coords[coord_name]
        if coord.dims:
            coords[coord_name] = (coord.dims, coord.values)
    return xr.Dataset(data_vars={var: (("time",) + spatial_dims, values)}, coords=coords)


def fill_missing_2d(values: np.ndarray) -> np.ndarray:
    filled = values.copy()
    for feature in range(filled.shape[1]):
        feature_values = filled[:, feature]
        mask = np.isfinite(feature_values)
        if not np.any(mask):
            continue
        feature_values[~mask] = np.nanmedian(feature_values[mask])
        filled[:, feature] = feature_values
    return filled


def unpack_predict_result(result):
    if isinstance(result, tuple):
        return tuple(np.asarray(item) for item in result)
    if isinstance(result, list):
        return tuple(np.asarray(item) for item in result)
    if hasattr(result, "keys") and "Z1" in result:
        z1 = np.asarray(result["Z1"])
        z0 = np.asarray(result["Z0"]) if "Z0" in result else None
        return z1, z0
    raise TypeError(f"Unsupported SBCK predict result type: {type(result)!r}")


def apply_sbck_mbcn(train_hist: dict, test_hist: dict, test_future: dict, simu: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import SBCK
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SBCK is not installed in this environment. Install the SBCK package to use --bc-method sbck_mbcn."
        ) from exc

    y0 = bundle_obs(train_hist)
    x0 = train_hist[simu]
    x1 = test_hist[simu]
    x2 = test_future[simu]

    if y0.ndim != 4 or y0.shape[-1] != 2:
        raise ValueError("sbck_mbcn currently expects paired bundles shaped (time, y, x, 2).")

    shape_train = y0.shape
    shape_hist = x1.shape
    shape_future = x2.shape

    y0_flat = y0.reshape(shape_train[0], -1, shape_train[-1])
    x0_flat = x0.reshape(shape_train[0], -1, shape_train[-1])
    x1_flat = x1.reshape(shape_hist[0], -1, shape_hist[-1])
    x2_flat = x2.reshape(shape_future[0], -1, shape_future[-1])

    z0_flat = np.full_like(x0_flat, np.nan, dtype=np.float64)
    z1_flat = np.full_like(x1_flat, np.nan, dtype=np.float64)
    z2_flat = np.full_like(x2_flat, np.nan, dtype=np.float64)

    for cell in range(y0_flat.shape[1]):
        y0_cell = y0_flat[:, cell, :]
        x0_cell = x0_flat[:, cell, :]
        x1_cell = x1_flat[:, cell, :]
        x2_cell = x2_flat[:, cell, :]

        train_mask = np.all(np.isfinite(y0_cell) & np.isfinite(x0_cell), axis=1)
        hist_mask = np.all(np.isfinite(x1_cell), axis=1)
        future_mask = np.all(np.isfinite(x2_cell), axis=1)
        if train_mask.sum() < 10 or hist_mask.sum() < 10:
            continue

        x0_fit = fill_missing_2d(x0_cell)
        x1_fit = fill_missing_2d(x1_cell)
        mbcn = SBCK.MBCn()
        mbcn.fit(y0_cell[train_mask], x0_cell[train_mask], x1_fit)

        z1_cell, z0_cell = unpack_predict_result(mbcn.predict(x1_fit, x0_fit))
        z1_flat[:, cell, :] = np.asarray(z1_cell).reshape(-1, x1_cell.shape[1])[: x1_cell.shape[0]]
        z0_flat[:, cell, :] = np.asarray(z0_cell).reshape(-1, x0_cell.shape[1])[: x0_cell.shape[0]]
        z1_flat[~hist_mask, cell, :] = np.nan
        z0_flat[~np.all(np.isfinite(x0_cell), axis=1), cell, :] = np.nan

        if future_mask.sum() > 0:
            x2_fit = fill_missing_2d(x2_cell)
            z2_cell = np.asarray(mbcn.predict(x2_fit)).reshape(-1, x2_cell.shape[1])[: x2_cell.shape[0]]
            z2_flat[:, cell, :] = z2_cell
            z2_flat[~future_mask, cell, :] = np.nan

    return (
        z0_flat.reshape(shape_train),
        z1_flat.reshape(shape_hist),
        z2_flat.reshape(shape_future),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bias correct paired variables jointly with SBCK MBCn")
    parser.add_argument("--exp", type=str, help="Experiment name (e.g., exp5)")
    parser.add_argument("--ssp", type=str, help="SSP scenario (e.g., ssp585)")
    parser.add_argument("--simu", type=str, help="Simulation alias or model source key", default="gcm")
    parser.add_argument("--var", type=str, default=None, help="Unused compatibility argument from the generic workflow runner.")
    parser.add_argument("--paired-vars", type=str, required=True, help="Exactly two comma-separated variables, e.g. uas,vas")
    parser.add_argument("--bc-tag", type=str, default=DEFAULT_BC_TAG, help="Optional suffix to keep BC outputs separate across methods.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variables = [item.strip() for item in args.paired_vars.split(",") if item.strip()]
    if len(variables) != 2:
        raise ValueError("--paired-vars requires exactly two variables.")

    exp = args.exp
    ssp = args.ssp
    simu = args.simu
    bc_tag = normalize_bc_tag(args.bc_tag) or DEFAULT_BC_TAG
    simu_source = get_simu_source(exp, simu)

    train_hist = dict(np.load(get_bc_bundle_path(exp, simu, "train_hist", variables=variables), allow_pickle=True))
    test_hist = dict(np.load(get_bc_bundle_path(exp, simu, "test_hist", variables=variables), allow_pickle=True))
    test_future = dict(np.load(get_bc_bundle_path(exp, simu, "test_future", variables=variables), allow_pickle=True))
    bundle_variables = [str(item) for item in train_hist.get("variables", [])]
    if bundle_variables and bundle_variables != variables:
        raise ValueError(f"Paired bundle variables {bundle_variables} do not match requested variables {variables}.")

    print("Applying SBCK MBCn", flush=True)
    train_hist_bc, test_hist_bc, test_future_bc = apply_sbck_mbcn(train_hist, test_hist, test_future, simu)

    geometry_source = corrected_geometry_reference(exp, simu, simu_source)
    get_data_bc = Data(CONFIG[exp].get("bc_domain", CONFIG[exp]["domain"]))
    reference_date = get_bc_train_hist_dates(exp)[0]
    for feature_index, var in enumerate(variables):
        model_ds = get_data_bc.get_model_dataset(geometry_source, var, reference_date, ssp=ssp)
        ds_train_hist_bc = None
        ds_test_hist_bc = None
        ds_test_future_bc = None
        try:
            ds_train_hist_bc = build_corrected_dataset(model_ds, var, train_hist_bc[..., feature_index], train_hist["dates"])
            ds_test_hist_bc = build_corrected_dataset(model_ds, var, test_hist_bc[..., feature_index], test_hist["dates"])
            ds_test_future_bc = build_corrected_dataset(model_ds, var, test_future_bc[..., feature_index], test_future["dates"])
            ds_train_hist_bc.to_netcdf(get_bias_corrected_netcdf_path(exp, simu, var, "train_hist", ssp=ssp, bc_tag=bc_tag))
            ds_test_hist_bc.to_netcdf(get_bias_corrected_netcdf_path(exp, simu, var, "test_hist", ssp=ssp, bc_tag=bc_tag))
            ds_test_future_bc.to_netcdf(get_bias_corrected_netcdf_path(exp, simu, var, "test_future", ssp=ssp, bc_tag=bc_tag))
        finally:
            model_ds.close()
            if ds_train_hist_bc is not None:
                ds_train_hist_bc.close()
            if ds_test_hist_bc is not None:
                ds_test_hist_bc.close()
            if ds_test_future_bc is not None:
                ds_test_future_bc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
