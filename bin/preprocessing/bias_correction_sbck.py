"""
Data correction, evaluation and saving of the bias corrected dataset using the
SBCK python library.

This script mirrors the production contract of bias_correction_ibicus.py:
- consumes bc_train_hist/test_hist/test_future_<simu>.npz
- writes canonical bias-corrected NetCDF outputs
- materializes dataset_<exp>_test_<simu>_bc sample files
"""

import sys
sys.path.append('.')

import argparse
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from ibicus.evaluate import marginal, metrics, trend

from iriscc.datautils import Data, reformat_as_target
from iriscc.settings import (
    CONFIG,
    DATASET_BC_DIR,
    DATES_BC_TEST_FUTURE,
    DATES_BC_TEST_HIST,
    DATES_BC_TRAIN_HIST,
    GRAPHS_DIR,
    get_bias_corrected_netcdf_path,
)


def plot_tprofiles_short_range(
    y: Optional[np.ndarray],
    x: np.ndarray,
    z: np.ndarray,
    title: str,
    savedir: str,
    simu: Optional[str] = None,
) -> None:
    plt.figure(figsize=(15, 5))
    if y is not None:
        plt.plot(y[1000:2000], label='ERA5', color='red')
    plt.plot(x[1000:2000], label=f'{simu}', color='blue')
    plt.plot(z[1000:2000], label=f'{simu} bc', color='green')
    plt.xlabel('Days')
    plt.ylabel('Daily {var}')
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
    simu: Optional[str] = None,
) -> None:
    i_summer = []
    i_winter = []
    for index, date in enumerate(pd.DatetimeIndex(dates)):
        if date.month in [3, 4, 5, 6, 7, 8]:
            i_summer.append(index)
        else:
            i_winter.append(index)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey='row')
    if y is not None:
        ax1.hist([y[i] for i in i_winter], histtype='step', color='red', label='ERA5', density=True)
        ax2.hist([y[i] for i in i_summer], histtype='step', color='red', label='ERA5', density=True)
    ax1.hist([x[i] for i in i_winter], histtype='step', color='blue', label=f'{simu}', density=True, range=(265, 300))
    ax1.hist([z[i] for i in i_winter], histtype='step', color='green', label=f'{simu} bc', density=True)
    ax2.hist([x[i] for i in i_summer], histtype='step', color='blue', label=f'{simu}', density=True, range=(265, 300))
    ax2.hist([z[i] for i in i_summer], histtype='step', color='green', label=f'{simu} bc', density=True)
    plt.suptitle(title)
    ax1.set_title('Winter')
    ax2.set_title('Summer')
    ax1.set_xlabel('{var} (K)')
    ax2.set_xlabel('{var} (K)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(savedir)


def monthly_mean(
    y: Optional[np.ndarray],
    x: np.ndarray,
    z: np.ndarray,
    dates: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if y is None:
        y = x
    df = pd.DataFrame({'dates': dates, 'y': y, 'x': x, 'z': z})
    df['month'] = pd.to_datetime(df['dates']).dt.month
    df['year'] = pd.to_datetime(df['dates']).dt.year
    df_month = df.groupby(['year', 'month']).mean().reset_index()
    y_month = df_month['y'].values
    x_month = df_month['x'].values
    z_month = df_month['z'].values
    dates_month = df_month['dates'].values
    return y_month, x_month, z_month, dates_month


def apply_sbck_cdft(train_hist: dict, test_hist: dict, test_future: dict, simu: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import SBCK
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SBCK is not installed in this environment. Install the SBCK package to use --bc-method sbck_cdft."
        ) from exc

    y0 = train_hist['era5']
    x0 = train_hist[simu]
    x1 = test_hist[simu]
    x2 = test_future[simu]

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bias correct and plot results with SBCK CDFt")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')
    parser.add_argument('--ssp', type=str, help='SSP scenario (e.g., ssp585)')
    parser.add_argument('--simu', type=str, help='Simulation alias or model source key', default='gcm')
    parser.add_argument('--var', type=str, help='tas, pr', default='tas')
    parser.add_argument('--test', action='store_true', help='Skip expensive diagnostics and only materialize corrected outputs')
    args = parser.parse_args()

    exp = args.exp
    ssp = args.ssp
    simu = args.simu
    var = args.var
    domain = CONFIG[exp]['domain']
    bc_domain = CONFIG[exp].get('bc_domain', [-12.5, 27.5, 31., 71.])
    gcm_source = CONFIG[exp].get('gcm_source', 'gcm_cnrm_cm6_1')
    orog_file = CONFIG[exp]['orog_file']
    target_file = CONFIG[exp]['target_file']
    dataset_bc_dir = DATASET_BC_DIR / f'dataset_{exp}_test_{simu}_bc'
    graphs_bias_dir = GRAPHS_DIR / 'biascorrection'
    dataset_bc_dir.mkdir(parents=True, exist_ok=True)
    graphs_bias_dir.mkdir(parents=True, exist_ok=True)

    get_data_bc = Data(bc_domain)
    gcm_ds = get_data_bc.get_model_dataset(gcm_source, var, DATES_BC_TRAIN_HIST[0], ssp=ssp)
    lon, lat = gcm_ds.lon.values, gcm_ds.lat.values
    get_data = Data(domain=domain)

    train_hist = dict(np.load(DATASET_BC_DIR / f'bc_train_hist_{simu}.npz', allow_pickle=True))
    test_hist = dict(np.load(DATASET_BC_DIR / f'bc_test_hist_{simu}.npz', allow_pickle=True))
    test_future = dict(np.load(DATASET_BC_DIR / f'bc_test_future_{simu}.npz', allow_pickle=True))

    print("Applying SBCK CDFt", flush=True)
    train_hist_bc, test_hist_bc, test_future_bc = apply_sbck_cdft(train_hist, test_hist, test_future, simu)

    if args.test:
        print("Skipping SBCK diagnostics in --test mode", flush=True)
    else:
        var_marginal_bias_data = marginal.calculate_marginal_bias(
            metrics=[metrics.cold_days, metrics.warm_days],
            percentage_or_absolute='absolute',
            obs=test_hist['era5'],
            raw=test_hist[simu],
            CDFt=test_hist_bc,
        )
        plot = marginal.plot_marginal_bias(variable=var, bias_df=var_marginal_bias_data)
        plot.savefig(GRAPHS_DIR / f'biascorrection/{var}_sbck_bias_boxplot_{simu}.png')

        var_trend_bias_data = trend.calculate_future_trend_bias(
            statistics=["mean", 0.05, 0.95],
            trend_type='additive',
            raw_validate=test_hist[simu],
            raw_future=test_future[simu],
            metrics=[metrics.cold_days, metrics.warm_days],
            CDFt=[test_hist_bc, test_future_bc],
        )

        plot = trend.plot_future_trend_bias_boxplot(variable=var, bias_df=var_trend_bias_data, remove_outliers=True)
        plot.savefig(GRAPHS_DIR / f'biascorrection/{var}_sbck_bias_futur_trend_{ssp}_{simu}.png')

        y0_mean = np.mean(train_hist['era5'], axis=(1, 2))
        x0_mean = np.mean(train_hist[simu], axis=(1, 2))
        z0_mean = np.mean(train_hist_bc, axis=(1, 2))
        y1_mean = np.mean(test_hist['era5'], axis=(1, 2))
        x1_mean = np.mean(test_hist[simu], axis=(1, 2))
        z1_mean = np.mean(test_hist_bc, axis=(1, 2))
        x2_mean = np.mean(test_future[simu], axis=(1, 2))
        z2_mean = np.mean(test_future_bc, axis=(1, 2))

        plot_tprofiles_short_range(
            y0_mean,
            x0_mean,
            z0_mean,
            title=f'Daily {var} over the historical Train period (1980-1999)',
            savedir=GRAPHS_DIR / f'biascorrection/{var}_sbck_train_hist_tprofiles_{simu}.png',
            simu=simu,
        )
        plot_tprofiles_short_range(
            y1_mean,
            x1_mean,
            z1_mean,
            title=f'Daily {var} over the historical Test period (2000-2014)',
            savedir=GRAPHS_DIR / f'biascorrection/{var}_sbck_test_hist_tprofiles_{simu}.png',
            simu=simu,
        )
        plot_tprofiles_short_range(
            None,
            x2_mean,
            z2_mean,
            title=f'Daily {var} over the future Test period (2015-2100 {ssp})',
            savedir=GRAPHS_DIR / f'biascorrection/{var}_sbck_test_future_tprofiles_{ssp}_{simu}.png',
            simu=simu,
        )

        plt.figure(figsize=(6, 6))
        plt.scatter(x1_mean, z1_mean, color='blue', s=5, label='2000-2014')
        plt.scatter(x2_mean, z2_mean, color='green', s=5, label=f'2015-2100 {ssp}')
        plt.plot(np.arange(270, 300), np.arange(270, 300), color='black')
        plt.xlim(270, 300)
        plt.ylim(270, 300)
        plt.legend()
        plt.xlabel(f'{var} {simu} (K)')
        plt.ylabel(f'{var} {simu} bc (K)')
        plt.title(f'Daily mean {var} over the historical and future test period')
        plt.savefig(GRAPHS_DIR / f'biascorrection/{var}_sbck_test_hist_linear_{ssp}_{simu}.png')

        df_simu = pd.DataFrame({
            'dates': np.concatenate((train_hist['dates'], test_hist['dates'], test_future['dates']), axis=None),
            'values': np.concatenate((x0_mean, x1_mean, x2_mean), axis=None),
            'labels': np.concatenate((np.ones_like(x0_mean), 2 * np.ones_like(x1_mean), 3 * np.ones_like(x2_mean)), axis=None),
        })
        df_simu['year'] = pd.to_datetime(df_simu['dates']).dt.year
        df_simu_year = df_simu.groupby('year').mean()

        df_era5 = pd.DataFrame({
            'dates': np.concatenate((train_hist['dates'], test_hist['dates']), axis=None),
            'values': np.concatenate((y0_mean, y1_mean), axis=None),
            'labels': np.concatenate((np.ones_like(y0_mean), 2 * np.ones_like(y1_mean)), axis=None),
        })
        df_era5['year'] = pd.to_datetime(df_era5['dates']).dt.year
        df_era5_year = df_era5.groupby('year').mean()

        df_simu_bc = pd.DataFrame({
            'dates': np.concatenate((train_hist['dates'], test_hist['dates'], test_future['dates']), axis=None),
            'values': np.concatenate((z0_mean, z1_mean, z2_mean), axis=None),
            'labels': np.concatenate((np.ones_like(z0_mean), 2 * np.ones_like(z1_mean), 3 * np.ones_like(z2_mean)), axis=None),
        })
        df_simu_bc['year'] = pd.to_datetime(df_simu_bc['dates']).dt.year
        df_simu_bc_year = df_simu_bc.groupby('year').mean()

        plt.figure(figsize=(8, 4))
        plt.plot(df_era5_year.index, np.where(df_era5_year["labels"] == 1., df_era5_year["values"], None), color="red", label='ERA5')
        plt.plot(df_simu_year.index, np.where(df_simu_year["labels"] == 1., df_simu_year["values"], None), color="blue", label=simu)
        plt.plot(df_simu_bc_year.index, np.where(df_simu_bc_year["labels"] == 1., df_simu_bc_year["values"], None), color="green", label=f'{simu} bc')
        plt.plot(df_era5_year.index, np.where(df_era5_year["labels"] == 2., df_era5_year["values"], None), color="red")
        plt.plot(df_simu_year.index, np.where(df_simu_year["labels"] == 2., df_simu_year["values"], None), color="blue")
        plt.plot(df_simu_bc_year.index, np.where(df_simu_bc_year["labels"] == 2., df_simu_bc_year["values"], None), color="green")
        plt.plot(df_simu_year.index, np.where(df_simu_year["labels"] == 3., df_simu_year["values"], None), color="blue")
        plt.plot(df_simu_bc_year.index, np.where(df_simu_bc_year["labels"] == 3., df_simu_bc_year["values"], None), color="green")
        plt.title(f'Annual mean {var} {simu} ({ssp})')
        plt.ylabel('{var} (K)')
        plt.legend()
        plt.savefig(GRAPHS_DIR / f'biascorrection/{var}_bc_datasets_temporal_profiles_sbck_{ssp}_{simu}.png')

        y0_hist, x0_hist, z0_hist, dates = monthly_mean(y0_mean, x0_mean, z0_mean, train_hist['dates'])
        plot_seasonal_hist(
            y0_hist,
            x0_hist,
            z0_hist,
            dates,
            title=f'Monthly mean {var} over the historical Train period (1980-1999)',
            savedir=GRAPHS_DIR / f'biascorrection/{var}_sbck_train_hist_histo_{simu}.png',
            simu=simu,
        )
        y1_hist, x1_hist, z1_hist, dates = monthly_mean(y1_mean, x1_mean, z1_mean, test_hist['dates'])
        plot_seasonal_hist(
            y1_hist,
            x1_hist,
            z1_hist,
            dates,
            title=f'Monthly mean {var} over the historical Test period (2000-2014)',
            savedir=GRAPHS_DIR / f'biascorrection/{var}_sbck_test_hist_histo_{simu}.png',
            simu=simu,
        )
        _, x2_hist, z2_hist, dates = monthly_mean(None, x2_mean, z2_mean, test_future['dates'])
        plot_seasonal_hist(
            None,
            x2_hist,
            z2_hist,
            dates,
            title=f'Monthly mean {var} over the future Test period (2015-2100 {ssp})',
            savedir=GRAPHS_DIR / f'biascorrection/{var}_sbck_test_future_histo_{ssp}_{simu}.png',
            simu=simu,
        )

    ds_train_hist_bc = xr.Dataset(
        data_vars=dict(tas=(['time', 'lat', 'lon'], train_hist_bc)),
        coords=dict(lat=('lat', lat), lon=('lon', lon), time=('time', train_hist['dates'])),
    )
    ds_test_hist_bc = xr.Dataset(
        data_vars=dict(tas=(['time', 'lat', 'lon'], test_hist_bc)),
        coords=dict(lat=('lat', lat), lon=('lon', lon), time=('time', test_hist['dates'])),
    )
    ds_test_future_bc = xr.Dataset(
        data_vars=dict(tas=(['time', 'lat', 'lon'], test_future_bc)),
        coords=dict(lat=('lat', lat), lon=('lon', lon), time=('time', test_future['dates'])),
    )
    ds_train_hist_bc.to_netcdf(get_bias_corrected_netcdf_path(exp, simu, var, 'train_hist', ssp=ssp))
    ds_test_hist_bc.to_netcdf(get_bias_corrected_netcdf_path(exp, simu, var, 'test_hist', ssp=ssp))
    ds_test_future_bc.to_netcdf(get_bias_corrected_netcdf_path(exp, simu, var, 'test_future', ssp=ssp))

    for date in pd.DatetimeIndex(ds_train_hist_bc.time.values):
        print(date)
        x = []

        ds = xr.open_dataset(orog_file)
        x.append(ds['elevation'].values)
        ds_train_hist_bc_i = ds_train_hist_bc.sel(time=ds_train_hist_bc.time.dt.date == date.date()).isel(time=0, drop=True)
        ds_train_hist_bc_i = reformat_as_target(
            ds_train_hist_bc_i,
            target_file=target_file,
            domain=domain,
            method="conservative_normed",
            mask=True,
        )

        x.append(ds_train_hist_bc_i.tas.values)
        x = np.stack(x, axis=0)
        ds_target = get_data.get_target_dataset(
            target=CONFIG[exp]['target'],
            var=var,
            date=date,
            source_name=CONFIG[exp].get('target_source'),
        )
        y = np.expand_dims(ds_target[var].values, axis=0)

        sample = {'x': x, 'y': y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(dataset_bc_dir / f'sample_{date_str}.npz', **sample)

    for date in pd.DatetimeIndex(ds_test_hist_bc.time.values):
        print(date)
        x = []

        ds = xr.open_dataset(orog_file)
        x.append(ds['elevation'].values)
        ds_test_hist_bc_i = ds_test_hist_bc.sel(time=ds_test_hist_bc.time.dt.date == date.date()).isel(time=0, drop=True)
        ds_test_hist_bc_i = reformat_as_target(
            ds_test_hist_bc_i,
            target_file=target_file,
            domain=domain,
            method="conservative_normed",
            mask=True,
        )

        x.append(ds_test_hist_bc_i.tas.values)
        x = np.stack(x, axis=0)
        ds_target = get_data.get_target_dataset(
            target=CONFIG[exp]['target'],
            var=var,
            date=date,
            source_name=CONFIG[exp].get('target_source'),
        )
        y = np.expand_dims(ds_target[var].values, axis=0)

        sample = {'x': x, 'y': y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(dataset_bc_dir / f'sample_{date_str}.npz', **sample)

    for date in pd.DatetimeIndex(ds_test_future_bc.time.values):
        print(date)
        x = []

        ds = xr.open_dataset(orog_file)
        x.append(ds['elevation'].values)
        ds_test_future_bc_i = ds_test_future_bc.sel(time=ds_test_future_bc.time.dt.date == date.date()).isel(time=0, drop=True)
        ds_test_future_bc_i = reformat_as_target(
            ds_test_future_bc_i,
            target_file=target_file,
            domain=domain,
            method="conservative_normed",
            mask=True,
        )

        x.append(ds_test_future_bc_i.tas.values)
        x = np.stack(x, axis=0)

        sample = {'x': x}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(dataset_bc_dir / f'sample_{date_str}.npz', **sample)
