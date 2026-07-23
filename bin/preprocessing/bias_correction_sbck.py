"""
Data correction, evaluation and saving of the bias corrected dataset using the
SBCK python library.

This script mirrors the production definition of bias_correction_ibicus.py:
- consumes bc_train_hist/test_hist/test_future_<simu>.npz
- writes canonical bias-corrected NetCDF outputs
- materializes dataset_<exp>_test_<simu>_bc sample files
"""

import sys
sys.path.append(".")

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from ibicus.evaluate import marginal, metrics, trend

from bin.preprocessing.build_dataset_pp import build_perfect_model_target
from iriscc.datautils import Data, reformat_as_target, return_unit
from iriscc.settings import (
    CONFIG,
    GRAPHS_DIR,
    get_bc_bundle_path,
    get_bc_train_hist_dates,
    get_bias_corrected_sample_dir,
    get_bias_corrected_netcdf_path,
    get_simu_family,
    get_simu_source,
    get_source_output_label,
    normalize_bc_tag,
)


def plot_tprofiles_short_range(
    y: np.ndarray | None,
    x: np.ndarray,
    z: np.ndarray,
    title: str,
    savedir: str,
    var: str,
    simu: str | None = None,
) -> None:
    plt.figure(figsize=(15, 5))
    if y is not None:
        plt.plot(y[1000:2000], label="ERA5", color="red")
    plt.plot(x[1000:2000], label=f"{simu}", color="blue")
    plt.plot(z[1000:2000], label=f"{simu} bc", color="green")
    plt.xlabel("Days")
    plt.ylabel(f"Daily {var} ({return_unit(var)})")
    plt.title(title)
    plt.legend()
    plt.savefig(savedir)


def plot_seasonal_hist(
    y: list | None,
    x: list,
    z: list,
    dates: list,
    title: str,
    savedir: str,
    var: str,
    simu: str | None = None,
) -> None:
    i_summer = []
    i_winter = []
    for index, date in enumerate(pd.DatetimeIndex(dates)):
        if date.month in [3, 4, 5, 6, 7, 8]:
            i_summer.append(index)
        else:
            i_winter.append(index)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey="row")
    if y is not None:
        ax1.hist([y[i] for i in i_winter], histtype="step", color="red", label="ERA5", density=True)
        ax2.hist([y[i] for i in i_summer], histtype="step", color="red", label="ERA5", density=True)
    ax1.hist([x[i] for i in i_winter], histtype="step", color="blue", label=f"{simu}", density=True)
    ax1.hist([z[i] for i in i_winter], histtype="step", color="green", label=f"{simu} bc", density=True)
    ax2.hist([x[i] for i in i_summer], histtype="step", color="blue", label=f"{simu}", density=True)
    ax2.hist([z[i] for i in i_summer], histtype="step", color="green", label=f"{simu} bc", density=True)
    plt.suptitle(title)
    ax1.set_title("Winter")
    ax2.set_title("Summer")
    unit = return_unit(var)
    ax1.set_xlabel(f"{var} ({unit})")
    ax2.set_xlabel(f"{var} ({unit})")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(savedir)


def monthly_mean(
    y: np.ndarray | None,
    x: np.ndarray,
    z: np.ndarray,
    dates: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if y is None:
        y = x
    df = pd.DataFrame({"dates": dates, "y": y, "x": x, "z": z})
    df["month"] = pd.to_datetime(df["dates"]).dt.month
    df["year"] = pd.to_datetime(df["dates"]).dt.year
    df_month = df.groupby(["year", "month"]).mean().reset_index()
    y_month = df_month["y"].values
    x_month = df_month["x"].values
    z_month = df_month["z"].values
    dates_month = df_month["dates"].values
    return y_month, x_month, z_month, dates_month


def bundle_obs(bundle: dict) -> np.ndarray:
    if "obs" in bundle:
        return bundle["obs"]
    if "era5" in bundle:
        return bundle["era5"]
    raise KeyError("BC bundle does not contain an 'obs' or legacy 'era5' reference field.")


def validate_training_shapes(reference: np.ndarray, simulation: np.ndarray, simu: str) -> None:
    if reference.shape != simulation.shape:
        raise ValueError(
            "BC training reference and simulation arrays must have identical shapes before SBCK. "
            f"reference shape={reference.shape}, {simu} shape={simulation.shape}. "
            "Regenerate the bc_dataset bundle and check that the reference source is time-varying "
            "over the full training period."
        )


def target_sample_for_date(
    get_data: Data,
    exp: str,
    var: str,
    date,
    ssp: str,
    target_file,
    domain,
) -> np.ndarray:
    perfect_model_target_source = CONFIG[exp].get("perfect_model_target_source")
    if perfect_model_target_source:
        target_method = CONFIG[exp].get("perfect_model_target_method", "conservative_normed")
        y = build_perfect_model_target(
            get_data,
            perfect_model_target_source,
            var,
            date,
            ssp,
            target_file,
            domain,
            target_method,
        )
        return np.expand_dims(y, axis=0).astype(np.float32)

    ds_target = get_data.get_target_dataset(
        target=CONFIG[exp]["target"],
        var=var,
        date=date,
        source_name=CONFIG[exp].get("target_source"),
        skip_domain_crop=bool(CONFIG[exp].get("target_source_pregridded", False)),
    )
    try:
        return np.expand_dims(ds_target[var].values, axis=0).astype(np.float32)
    finally:
        ds_target.close()


def materialize_corrected_samples(
    *,
    corrected_ds: xr.Dataset,
    dates,
    dataset_bc_dir,
    orog: np.ndarray,
    target_file,
    domain,
    get_data: Data,
    exp: str,
    var: str,
    ssp: str,
    include_target: bool,
) -> None:
    for date in pd.DatetimeIndex(dates):
        print(date)
        ds_day = corrected_ds.sel(time=corrected_ds.time.dt.date == date.date()).isel(time=0, drop=True)
        ds_day_target = reformat_as_target(
            ds_day,
            target_file=target_file,
            domain=domain,
            method=CONFIG[exp].get("perfect_model_target_method", "conservative_normed"),
            mask=True,
        )
        try:
            x = np.stack([orog, ds_day_target[var].values], axis=0).astype(np.float32)
        finally:
            ds_day_target.close()

        sample = {"x": x}
        if include_target:
            sample["y"] = target_sample_for_date(
                get_data=get_data,
                exp=exp,
                var=var,
                date=date,
                ssp=ssp,
                target_file=target_file,
                domain=domain,
            )

        date_str = date.date().strftime("%Y%m%d")
        np.savez(dataset_bc_dir / f"sample_{date_str}.npz", **sample)


def apply_sbck_cdft(train_hist: dict, test_hist: dict, test_future: dict, simu: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import SBCK
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SBCK is not installed in this environment. Install the SBCK package to use --bc-method sbck_cdft."
        ) from exc

    y0 = bundle_obs(train_hist)
    x0 = train_hist[simu]
    x1 = test_hist[simu]
    x2 = test_future[simu]
    validate_training_shapes(y0, x0, simu)

    shape_train = y0.shape
    shape_hist = x1.shape
    shape_future = x2.shape

    y0_flat = y0.reshape(shape_train[0], -1)
    x0_flat = x0.reshape(shape_train[0], -1)
    x1_flat = x1.reshape(shape_hist[0], -1)
    x2_flat = x2.reshape(shape_future[0], -1)

    z0_flat = np.full_like(x0_flat, np.nan, dtype=np.float64)
    z1_flat = np.full_like(x1_flat, np.nan, dtype=np.float64)
    z2_flat = np.full_like(x2_flat, np.nan, dtype=np.float64)

    for cell in range(y0_flat.shape[1]):
        y0_cell = y0_flat[:, cell]
        x0_cell = x0_flat[:, cell]
        x1_cell = x1_flat[:, cell]
        x2_cell = x2_flat[:, cell]

        if np.all(np.isnan(y0_cell)) or np.all(np.isnan(x0_cell)):
            continue

        train_mask = np.isfinite(y0_cell) & np.isfinite(x0_cell)
        hist_mask = np.isfinite(x1_cell)
        future_mask = np.isfinite(x2_cell)
        if train_mask.sum() < 10 or hist_mask.sum() < 10:
            continue

        x1_fit = np.where(hist_mask, x1_cell, np.nanmedian(x1_cell[hist_mask]))
        cdft = SBCK.CDFt(version=2, normalize_cdf=True)
        cdft.fit(y0_cell[train_mask], x0_cell[train_mask], x1_fit)

        z1_cell, z0_cell = cdft.predict(x1_fit, x0_cell)
        z1_flat[:, cell] = np.asarray(z1_cell).reshape(-1)[: x1_cell.shape[0]]
        z0_flat[:, cell] = np.asarray(z0_cell).reshape(-1)[: x0_cell.shape[0]]
        z1_flat[~hist_mask, cell] = np.nan
        z0_flat[~np.isfinite(x0_cell), cell] = np.nan

        if future_mask.sum() > 0:
            x2_fit = np.where(future_mask, x2_cell, np.nanmedian(x2_cell[future_mask]))
            z2_cell, _ = cdft.predict(x2_fit, x0_cell)
            z2_flat[:, cell] = np.asarray(z2_cell).reshape(-1)[: x2_cell.shape[0]]
            z2_flat[~future_mask, cell] = np.nan

    return (
        z0_flat.reshape(shape_train),
        z1_flat.reshape(shape_hist),
        z2_flat.reshape(shape_future),
    )


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
    return xr.Dataset(
        data_vars=dict(**{var: (("time",) + spatial_dims, values)}),
        coords=coords,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bias correct and plot results with SBCK CDFt")
    parser.add_argument("--exp", type=str, help="Experiment name (e.g., exp1)")
    parser.add_argument("--ssp", type=str, help="SSP scenario (e.g., ssp585)")
    parser.add_argument("--simu", type=str, help="Simulation alias or model source key", default="gcm")
    parser.add_argument("--var", type=str, help="Scalar variable to bias-correct", default="tas")
    parser.add_argument("--test", action="store_true", help="Skip expensive diagnostics and only materialize corrected outputs")
    parser.add_argument("--bc-tag", type=str, default=None, help="Optional suffix to keep BC outputs separate across methods.")
    args = parser.parse_args()

    exp = args.exp
    ssp = args.ssp
    simu = args.simu
    var = args.var
    bc_tag = normalize_bc_tag(args.bc_tag)
    domain = CONFIG[exp]["domain"]
    bc_domain = CONFIG[exp].get("bc_domain", [-12.5, 27.5, 31., 71.])
    simu_source = get_simu_source(exp, simu)
    bc_reference_source = CONFIG[exp].get("bc_reanalysis_source", "era5")
    bc_reference_label = get_source_output_label(bc_reference_source)
    orog_file = CONFIG[exp]["orog_file"]
    target_file = CONFIG[exp]["target_file"]
    dataset_bc_dir = get_bias_corrected_sample_dir(exp, simu, bc_tag)
    graphs_bias_dir = GRAPHS_DIR / "biascorrection"
    dataset_bc_dir.mkdir(parents=True, exist_ok=True)
    graphs_bias_dir.mkdir(parents=True, exist_ok=True)

    get_data_bc = Data(bc_domain)
    geometry_source = corrected_geometry_reference(exp, simu, simu_source)
    model_ds = get_data_bc.get_model_dataset(geometry_source, var, get_bc_train_hist_dates(exp)[0], ssp=ssp)
    get_data = Data(domain=domain)

    train_hist = dict(np.load(get_bc_bundle_path(exp, simu, "train_hist"), allow_pickle=True))
    test_hist = dict(np.load(get_bc_bundle_path(exp, simu, "test_hist"), allow_pickle=True))
    test_future = dict(np.load(get_bc_bundle_path(exp, simu, "test_future"), allow_pickle=True))
    train_obs = bundle_obs(train_hist)
    test_obs = bundle_obs(test_hist)

    print("Applying SBCK CDFt", flush=True)
    train_hist_bc, test_hist_bc, test_future_bc = apply_sbck_cdft(train_hist, test_hist, test_future, simu)

    if args.test:
        print("Skipping SBCK diagnostics in --test mode", flush=True)
    else:
        var_marginal_bias_data = marginal.calculate_marginal_bias(
            metrics=[metrics.cold_days, metrics.warm_days],
            percentage_or_absolute="absolute",
            obs=test_obs,
            raw=test_hist[simu],
            CDFt=test_hist_bc,
        )
        plot = marginal.plot_marginal_bias(variable=var, bias_df=var_marginal_bias_data)
        plot.savefig(GRAPHS_DIR / f"biascorrection/{var}_sbck_bias_boxplot_{simu}.png")

        var_trend_bias_data = trend.calculate_future_trend_bias(
            statistics=["mean", 0.05, 0.95],
            trend_type="additive",
            raw_validate=test_hist[simu],
            raw_future=test_future[simu],
            metrics=[metrics.cold_days, metrics.warm_days],
            CDFt=[test_hist_bc, test_future_bc],
        )

        plot = trend.plot_future_trend_bias_boxplot(variable=var, bias_df=var_trend_bias_data, remove_outliers=True)
        plot.savefig(GRAPHS_DIR / f"biascorrection/{var}_sbck_bias_futur_trend_{ssp}_{simu}.png")

        y0_mean = np.mean(train_obs, axis=(1, 2))
        x0_mean = np.mean(train_hist[simu], axis=(1, 2))
        z0_mean = np.mean(train_hist_bc, axis=(1, 2))
        y1_mean = np.mean(test_obs, axis=(1, 2))
        x1_mean = np.mean(test_hist[simu], axis=(1, 2))
        z1_mean = np.mean(test_hist_bc, axis=(1, 2))
        x2_mean = np.mean(test_future[simu], axis=(1, 2))
        z2_mean = np.mean(test_future_bc, axis=(1, 2))

        plot_tprofiles_short_range(
            y0_mean,
            x0_mean,
            z0_mean,
            title=f"Daily {var} over the historical Train period (1980-1999)",
            savedir=GRAPHS_DIR / f"biascorrection/{var}_sbck_train_hist_tprofiles_{simu}.png",
            var=var,
            simu=simu,
        )
        plot_tprofiles_short_range(
            y1_mean,
            x1_mean,
            z1_mean,
            title=f"Daily {var} over the historical Test period (2000-2014)",
            savedir=GRAPHS_DIR / f"biascorrection/{var}_sbck_test_hist_tprofiles_{simu}.png",
            var=var,
            simu=simu,
        )
        plot_tprofiles_short_range(
            None,
            x2_mean,
            z2_mean,
            title=f"Daily {var} over the future Test period (2015-2100 {ssp})",
            savedir=GRAPHS_DIR / f"biascorrection/{var}_sbck_test_future_tprofiles_{ssp}_{simu}.png",
            var=var,
            simu=simu,
        )

        plt.figure(figsize=(6, 6))
        plt.scatter(x1_mean, z1_mean, color="blue", s=5, label="2000-2014")
        plt.scatter(x2_mean, z2_mean, color="green", s=5, label=f"2015-2100 {ssp}")
        plt.plot(np.arange(270, 300), np.arange(270, 300), color="black")
        plt.legend()
        unit = return_unit(var)
        plt.xlabel(f"{var} {simu} ({unit})")
        plt.ylabel(f"{var} {simu} bc ({unit})")
        plt.title(f"Daily mean {var} over the historical and future test period")
        plt.savefig(GRAPHS_DIR / f"biascorrection/{var}_sbck_test_hist_linear_{ssp}_{simu}.png")

        df_simu = pd.DataFrame({
            "dates": np.concatenate((train_hist["dates"], test_hist["dates"], test_future["dates"]), axis=None),
            "values": np.concatenate((x0_mean, x1_mean, x2_mean), axis=None),
            "labels": np.concatenate((np.ones_like(x0_mean), 2 * np.ones_like(x1_mean), 3 * np.ones_like(x2_mean)), axis=None),
        })
        df_simu["year"] = pd.to_datetime(df_simu["dates"]).dt.year
        df_simu_year = df_simu.groupby("year").mean()

        df_obs = pd.DataFrame({
            "dates": np.concatenate((train_hist["dates"], test_hist["dates"]), axis=None),
            "values": np.concatenate((y0_mean, y1_mean), axis=None),
            "labels": np.concatenate((np.ones_like(y0_mean), 2 * np.ones_like(y1_mean)), axis=None),
        })
        df_obs["year"] = pd.to_datetime(df_obs["dates"]).dt.year
        df_obs_year = df_obs.groupby("year").mean()

        df_simu_bc = pd.DataFrame({
            "dates": np.concatenate((train_hist["dates"], test_hist["dates"], test_future["dates"]), axis=None),
            "values": np.concatenate((z0_mean, z1_mean, z2_mean), axis=None),
            "labels": np.concatenate((np.ones_like(z0_mean), 2 * np.ones_like(z1_mean), 3 * np.ones_like(z2_mean)), axis=None),
        })
        df_simu_bc["year"] = pd.to_datetime(df_simu_bc["dates"]).dt.year
        df_simu_bc_year = df_simu_bc.groupby("year").mean()

        plt.figure(figsize=(8, 4))
        plt.plot(df_obs_year.index, np.where(df_obs_year["labels"] == 1., df_obs_year["values"], None), color="red", label=bc_reference_label)
        plt.plot(df_simu_year.index, np.where(df_simu_year["labels"] == 1., df_simu_year["values"], None), color="blue", label=simu)
        plt.plot(df_simu_bc_year.index, np.where(df_simu_bc_year["labels"] == 1., df_simu_bc_year["values"], None), color="green", label=f"{simu} bc")
        plt.plot(df_obs_year.index, np.where(df_obs_year["labels"] == 2., df_obs_year["values"], None), color="red")
        plt.plot(df_simu_year.index, np.where(df_simu_year["labels"] == 2., df_simu_year["values"], None), color="blue")
        plt.plot(df_simu_bc_year.index, np.where(df_simu_bc_year["labels"] == 2., df_simu_bc_year["values"], None), color="green")
        plt.plot(df_simu_year.index, np.where(df_simu_year["labels"] == 3., df_simu_year["values"], None), color="blue")
        plt.plot(df_simu_bc_year.index, np.where(df_simu_bc_year["labels"] == 3., df_simu_bc_year["values"], None), color="green")
        plt.title(f"Annual mean {var} {simu} ({ssp})")
        plt.ylabel(f"{var} ({return_unit(var)})")
        plt.legend()
        plt.savefig(GRAPHS_DIR / f"biascorrection/{var}_bc_datasets_temporal_profiles_sbck_{ssp}_{simu}.png")

        y0_hist, x0_hist, z0_hist, dates = monthly_mean(y0_mean, x0_mean, z0_mean, train_hist["dates"])
        plot_seasonal_hist(
            y0_hist,
            x0_hist,
            z0_hist,
            dates,
            title=f"Monthly mean {var} over the historical Train period (1980-1999)",
            savedir=GRAPHS_DIR / f"biascorrection/{var}_sbck_train_hist_histo_{simu}.png",
            var=var,
            simu=simu,
        )
        y1_hist, x1_hist, z1_hist, dates = monthly_mean(y1_mean, x1_mean, z1_mean, test_hist["dates"])
        plot_seasonal_hist(
            y1_hist,
            x1_hist,
            z1_hist,
            dates,
            title=f"Monthly mean {var} over the historical Test period (2000-2014)",
            savedir=GRAPHS_DIR / f"biascorrection/{var}_sbck_test_hist_histo_{simu}.png",
            var=var,
            simu=simu,
        )
        _, x2_hist, z2_hist, dates = monthly_mean(None, x2_mean, z2_mean, test_future["dates"])
        plot_seasonal_hist(
            None,
            x2_hist,
            z2_hist,
            dates,
            title=f"Monthly mean {var} over the future Test period (2015-2100 {ssp})",
            savedir=GRAPHS_DIR / f"biascorrection/{var}_sbck_test_future_histo_{ssp}_{simu}.png",
            var=var,
            simu=simu,
        )

    ds_train_hist_bc = build_corrected_dataset(model_ds, var, train_hist_bc, train_hist["dates"])
    ds_test_hist_bc = build_corrected_dataset(model_ds, var, test_hist_bc, test_hist["dates"])
    ds_test_future_bc = build_corrected_dataset(model_ds, var, test_future_bc, test_future["dates"])
    ds_train_hist_bc.to_netcdf(get_bias_corrected_netcdf_path(exp, simu, var, "train_hist", ssp=ssp, bc_tag=bc_tag))
    ds_test_hist_bc.to_netcdf(get_bias_corrected_netcdf_path(exp, simu, var, "test_hist", ssp=ssp, bc_tag=bc_tag))
    ds_test_future_bc.to_netcdf(get_bias_corrected_netcdf_path(exp, simu, var, "test_future", ssp=ssp, bc_tag=bc_tag))
    with xr.open_dataset(orog_file) as ds_orog:
        orog = ds_orog["elevation"].values.astype(np.float32)

    materialize_corrected_samples(
        corrected_ds=ds_train_hist_bc,
        dates=ds_train_hist_bc.time.values,
        dataset_bc_dir=dataset_bc_dir,
        orog=orog,
        target_file=target_file,
        domain=domain,
        get_data=get_data,
        exp=exp,
        var=var,
        ssp=ssp,
        include_target=True,
    )
    materialize_corrected_samples(
        corrected_ds=ds_test_hist_bc,
        dates=ds_test_hist_bc.time.values,
        dataset_bc_dir=dataset_bc_dir,
        orog=orog,
        target_file=target_file,
        domain=domain,
        get_data=get_data,
        exp=exp,
        var=var,
        ssp=ssp,
        include_target=True,
    )
    materialize_corrected_samples(
        corrected_ds=ds_test_future_bc,
        dates=ds_test_future_bc.time.values,
        dataset_bc_dir=dataset_bc_dir,
        orog=orog,
        target_file=target_file,
        domain=domain,
        get_data=get_data,
        exp=exp,
        var=var,
        ssp=ssp,
        include_target=False,
    )
    ds_train_hist_bc.close()
    ds_test_hist_bc.close()
    ds_test_future_bc.close()
