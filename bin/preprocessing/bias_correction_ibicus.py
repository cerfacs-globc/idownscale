'''
Data correction, evaluation and saving of the bias corrected dataset using IBICUS python librairy

date : 16/07/2025
author : Zoé GARCIA
'''

import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from ibicus.evaluate import marginal, metrics, trend
from ibicus.debias import CDFt


from iriscc.datautils import (reformat_as_target, 
                              Data)
from iriscc.settings import (GRAPHS_DIR,
                             CONFIG,
                             DATES_BC_TEST_FUTURE,
                             DATES_BC_TEST_HIST,
                             DATES_BC_TRAIN_HIST,
                             DATASET_BC_DIR,
                             GCM_RAW_DIR,
                             RCM_RAW_DIR)


def plot_tprofiles_short_range(
        y: Optional[np.ndarray], 
        x: np.ndarray, 
        z: np.ndarray,
        title: str, 
        savedir: str,
        simu: Optional[str] = None
    ) -> None:
    """
    Plots {var} profiles for a short range of days and saves the plot to a file.
    """
    plt.figure(figsize=(15, 5))
    if y is not None:
        plt.plot(y[1000:2000],label='ERA5', color='red')
    plt.plot(x[1000:2000],label=f'{simu}', color='blue')
    plt.plot(z[1000:2000],label=f'{simu} bc', color='green')
    plt.xlabel('Days')
    plt.ylabel('Daily {var} ')
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
        simu: Optional[str] = None
    ) -> None:
    """
    Plots seasonal histograms (winter and summer) for {var} data.
    """
    i_summer = []
    i_winter = []
    for index, date in enumerate(pd.DatetimeIndex(dates)):
        if date.month in [3,4,5,6,7,8]:
            i_summer.append(index)
        else:
            i_winter.append(index)
    _, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4), sharey='row')
    if y is not None:
        ax1.hist([y[i] for i in i_winter], histtype='step', color='red', label='ERA5', density=True)
        ax2.hist([y[i] for i in i_summer], histtype='step', color='red', label='ERA5', density=True)
    ax1.hist([x[i] for i in i_winter], histtype='step', color='blue', label=f'{simu}', density=True, range=(265,300))
    ax1.hist([z[i] for i in i_winter], histtype='step', color='green', label=f'{simu} bc', density=True)
    ax2.hist([x[i] for i in i_summer], histtype='step', color='blue', label=f'{simu}', density=True, range=(265,300))
    ax2.hist([z[i] for i in i_summer], histtype='step', color='green', label=f'{simu} bc', density=True)
    plt.suptitle(title)
    ax1.set_title('Winter')
    ax2.set_title('Summer')
    ax1.set_xlabel('{var} (K)')
    ax2.set_xlabel('{var} (K)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(savedir)
    
def monthly_mean(y: Optional[np.ndarray], 
                 x: np.ndarray, 
                 z: np.ndarray, 
                 dates: List[str]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the monthly mean of input arrays grouped by year and month.
    """
    if y is None:
        y_vals = x
    else:
        y_vals = y
    df = pd.DataFrame({'dates' : dates,
                                'y': y_vals,
                                'x' : x,
                                'z' : z})
    df['month'] = pd.to_datetime(df['dates']).dt.month
    df['year'] = pd.to_datetime(df['dates']).dt.year
    df_month = df.groupby(['year','month']).mean().reset_index()
    y_month = df_month['y'].values
    x_month = df_month['x'].values
    z_month = df_month['z'].values
    dates_month = df_month['dates'].values
    return y_month, x_month, z_month, dates_month


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Predict and plot results (v86.74 Hardened)")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp5', default='exp5') 
    parser.add_argument('--ssp', type=str, help='SSP scenario (e.g., ssp585)', default='ssp585')
    parser.add_argument('--simu', type=str, help='gcm or rcm', default='gcm')
    parser.add_argument('--var', type=str, help='tas, pr', default='tas')
    args = parser.parse_args()

    exp = args.exp
    ssp = args.ssp
    simu = args.simu
    var = args.var
    domain =  CONFIG[exp]['domain']
    orog_file = CONFIG[exp]['orog_file']
    target_file = CONFIG[exp]['target_file']
    
    # Industrial Sanitization Protocol (v86.74)
    (GRAPHS_DIR / 'biascorrection').mkdir(parents=True, exist_ok=True)
    (DATASET_BC_DIR / f'dataset_{exp}_test_{simu}_bc').mkdir(parents=True, exist_ok=True)

    # 1. Loading Isolated Periods
    train_hist = dict(np.load(DATASET_BC_DIR/f'bc_train_hist_{simu}.npz', allow_pickle=True))
    test_hist = dict(np.load(DATASET_BC_DIR/f'bc_test_hist_{simu}.npz', allow_pickle=True))
    test_future = dict(np.load(DATASET_BC_DIR/f'bc_test_future_{simu}.npz', allow_pickle=True))

    debiaser = CDFt.from_variable(variable=var, running_window_mode=True)

    # 2. Debiasing Blocks
    print("--- Applying CDF-t (Historical Train) ---", flush=True)
    train_hist_bc = debiaser.apply(obs=train_hist['era5'],
                                cm_hist=train_hist[simu], 
                                cm_future=train_hist[simu], 
                                time_obs=train_hist['dates'], 
                                time_cm_hist=train_hist['dates'],
                                time_cm_future=train_hist['dates'])

    print("--- Applying CDF-t (Historical Test) ---", flush=True)
    test_hist_bc = debiaser.apply(obs=train_hist['era5'],
                                cm_hist=train_hist[simu], 
                                cm_future=test_hist[simu], 
                                time_obs=train_hist['dates'], 
                                time_cm_hist=train_hist['dates'],
                                time_cm_future=test_hist['dates'])
    
    print("--- Applying CDF-t (Future Test) ---", flush=True)
    test_future_bc = debiaser.apply(obs=train_hist['era5'],
                            cm_hist=train_hist[simu], 
                            cm_future=test_future[simu], 
                            time_obs=train_hist['dates'], 
                            time_cm_hist=train_hist['dates'],
                            time_cm_future=test_future['dates'])
    
    # 3. Validation Metrics
    print("--- Computing Ibicus Metrics ---", flush=True)
    var_marginal_bias_data = marginal.calculate_marginal_bias(metrics = [metrics.cold_days, metrics.warm_days], 
                                                            percentage_or_absolute='absolute',
                                                            obs = test_hist['era5'],
                                                            raw = test_hist[simu],
                                                            CDFt = test_hist_bc)
    plot = marginal.plot_marginal_bias(variable = var, bias_df = var_marginal_bias_data)
    plot.savefig(GRAPHS_DIR /f'biascorrection/{var}_ibicus_bias_boxplot_{simu}.png')
    
    var_trend_bias_data = trend.calculate_future_trend_bias(statistics = ["mean", 0.05, 0.95], 
                                                        trend_type = 'additive',
                                                        raw_validate = test_hist[simu], raw_future = test_future[simu],
                                                        metrics = [metrics.cold_days, metrics.warm_days],
                                                        CDFt = [test_hist_bc, test_future_bc])

    plot = trend.plot_future_trend_bias_boxplot(variable =var, bias_df = var_trend_bias_data, remove_outliers = True)
    plot.savefig(GRAPHS_DIR / f'biascorrection/{var}_ibicus_bias_futur_trend_{ssp}_{simu}.png')

    # Temporal Aggregates
    Y0 = np.mean(train_hist['era5'], axis=(1,2))
    X0 = np.mean(train_hist[simu], axis=(1,2))
    Z0 = np.mean(train_hist_bc, axis=(1,2))
    Y1 = np.mean(test_hist['era5'], axis=(1,2))
    X1 = np.mean(test_hist[simu], axis=(1,2))
    Z1 = np.mean(test_hist_bc, axis=(1,2))
    X2 = np.mean(test_future[simu], axis=(1,2))
    Z2 = np.mean(test_future_bc, axis=(1,2))

    # 4. Plots (Resilient to missing era5 keys)
    plot_tprofiles_short_range(Y0, X0, Z0, 
                               title = f'Daily {var} over the historical Train period (1980-1999)',
                               savedir=GRAPHS_DIR / f'biascorrection/{var}_ibicus_train_hist_tprofiles_{simu}.png',
                               simu = simu)
    plot_tprofiles_short_range(Y1, X1, Z1, 
                               title = f'Daily {var} over the historical Test period (2000-2014)',
                               savedir=GRAPHS_DIR / f'biascorrection/{var}_ibicus_test_hist_tprofiles_{simu}.png',
                               simu=simu)
    plot_tprofiles_short_range(None, X2, Z2, 
                               title = f'Daily {var} over the future Test period (2015-2100 {ssp})',
                               savedir=GRAPHS_DIR / f'biascorrection/{var}_ibicus_test_future_tprofiles_{ssp}_{simu}.png',
                               simu = simu)

    # Annual Mean Comparison
    df_simu = pd.DataFrame({'dates' : np.concatenate((train_hist['dates'], test_hist['dates'], test_future['dates']), axis=None),
                    'values' : np.concatenate((X0, X1, X2), axis=None),
                    'labels' : np.concatenate((np.ones_like(X0), 2*np.ones_like(X1), 3*np.ones_like(X2)), axis=None)})
    df_simu['year'] = pd.to_datetime(df_simu['dates']).dt.year
    df_simu_year = df_simu.groupby('year').mean()

    df_era5 = pd.DataFrame({'dates' : np.concatenate((train_hist['dates'], test_hist['dates']), axis=None),
                    'values' : np.concatenate((Y0, Y1), axis=None),
                    'labels' : np.concatenate((np.ones_like(Y0), 2*np.ones_like(Y1)), axis=None)})
    df_era5['year'] = pd.to_datetime(df_era5['dates']).dt.year
    df_era5_year = df_era5.groupby('year').mean()

    df_simu_bc = pd.DataFrame({'dates' : np.concatenate((train_hist['dates'], test_hist['dates'], test_future['dates']), axis=None),
                    'values' : np.concatenate((Z0, Z1, Z2), axis=None),
                    'labels' : np.concatenate((np.ones_like(Z0), 2*np.ones_like(Z1), 3*np.ones_like(Z2)), axis=None)})
    df_simu_bc['year'] = pd.to_datetime(df_simu_bc['dates']).dt.year
    df_simu_bc_year = df_simu_bc.groupby('year').mean()

    plt.figure(figsize=(8, 4))
    plt.plot(df_era5_year.index, np.where(df_era5_year["labels"]==1., df_era5_year["values"], None), color="red", label='ERA5')
    plt.plot(df_simu_year.index, np.where(df_simu_year["labels"]==1., df_simu_year["values"], None), color="blue", label=simu)
    plt.plot(df_simu_bc_year.index, np.where(df_simu_bc_year["labels"]==1., df_simu_bc_year["values"], None), color="green", label=simu)
    plt.plot(df_era5_year.index, np.where(df_era5_year["labels"]==2., df_era5_year["values"], None), color="red")
    plt.plot(df_simu_year.index, np.where(df_simu_year["labels"]==2., df_simu_year["values"], None), color="blue")
    plt.plot(df_simu_bc_year.index, np.where(df_simu_bc_year["labels"]==2., df_simu_bc_year["values"], None), color="green")
    plt.plot(df_simu_year.index, np.where(df_simu_year["labels"]==3., df_simu_year["values"], None), color="blue")
    plt.plot(df_simu_bc_year.index, np.where(df_simu_bc_year["labels"]==3., df_simu_bc_year["values"], None), color="green")
    plt.title(f'Annual mean {var} {simu} ({ssp})')
    plt.ylabel(f'{var} (K)')
    plt.legend()
    plt.savefig(GRAPHS_DIR/f'biascorrection/{var}_bc_datasets_temporal_profiles_ibicus_{ssp}_{simu}.png')

    # 5. NetCDF Restoration (GCM Grids)
    get_data_bc = Data([-12.5, 27.5, 31., 71.])
    gcm_ds = get_data_bc.get_gcm_dataset(var, None)
    lon, lat = gcm_ds.lon.values, gcm_ds.lat.values
    
    ds_train_hist_bc = xr.Dataset(data_vars=dict(tas=(['time', 'lat', 'lon'], train_hist_bc)),
                                  coords=dict(lat=('lat', lat), lon=('lon', lon), time=('time', train_hist['dates'])))
    ds_test_hist_bc = xr.Dataset(data_vars=dict(tas=(['time', 'lat', 'lon'], test_hist_bc)),
                                 coords=dict(lat=('lat', lat), lon=('lon', lon), time=('time', test_hist['dates'])))
    ds_test_future_bc = xr.Dataset(data_vars=dict(tas=(['time', 'lat', 'lon'], test_future_bc)),
                                   coords=dict(lat=('lat', lat), lon=('lon', lon), time=('time', test_future['dates'])))
    
    if simu == 'gcm':
        (GCM_RAW_DIR/'CNRM-CM6-1-BC').mkdir(parents=True, exist_ok=True)
        ds_train_hist_bc.to_netcdf(GCM_RAW_DIR/f'CNRM-CM6-1-BC/{var}_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc')
        ds_test_hist_bc.to_netcdf(GCM_RAW_DIR/f'CNRM-CM6-1-BC/{var}_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc')
        ds_test_future_bc.to_netcdf(GCM_RAW_DIR/f'CNRM-CM6-1-BC/{var}_day_CNRM-CM6-1_{ssp}_r1i1p1f2_gr_20150101-21001231_bc.nc')

    # 6. Discretization Bridging
    get_data = Data(domain=domain)
    
    for label, dates, ds_bc in [("train", DATES_BC_TRAIN_HIST, ds_train_hist_bc), 
                                ("test", DATES_BC_TEST_HIST, ds_test_hist_bc)]:
        print(f"--- Discretizing {label} ---", flush=True)
        for date in dates:
            ds_i = ds_bc.sel(time=ds_bc.time.dt.date == date.date()).isel(time=0, drop=True)
            ds_i = reformat_as_target(ds_i, target_file=target_file, domain=domain, method="conservative_normed", mask=True)
            
            x = []
            ds_orog = xr.open_dataset(orog_file)
            x.append(ds_orog['elevation'].values)
            x.append(ds_i.tas.values)
            
            ds_target = get_data.get_target_dataset(target=CONFIG[exp]['target'], var=var, date=date)
            y = np.expand_dims(ds_target[var].values, axis=0)
            
            date_str = date.date().strftime('%Y%m%d')
            np.savez_compressed(DATASET_BC_DIR/f'dataset_{exp}_test_{simu}_bc/sample_{date_str}.npz', x=np.stack(x, axis=0), y=y)

    # Future Projection
    print("--- Discretizing future ---", flush=True)
    for date in DATES_BC_TEST_FUTURE:
        ds_i = ds_test_future_bc.sel(time=ds_test_future_bc.time.dt.date == date.date()).isel(time=0, drop=True)
        ds_i = reformat_as_target(ds_i, target_file=target_file, domain=domain, method="conservative_normed", mask=True)
        
        x = []
        ds_orog = xr.open_dataset(orog_file)
        x.append(ds_orog['elevation'].values)
        x.append(ds_i.tas.values)
        
        date_str = date.date().strftime('%Y%m%d')
        np.savez_compressed(DATASET_BC_DIR/f'dataset_{exp}_test_{simu}_bc/sample_{date_str}.npz', x=np.stack(x, axis=0))