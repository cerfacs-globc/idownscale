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

    Args:
        y (Optional[np.ndarray]): {var} data for ERA5 (can be None).
        x (np.ndarray): {var} data for simulations.
        z (np.ndarray): Bias-corrected {var} data for simulations.
        title (str): Title of the plot.
        savedir (str): Path to save the generated plot.

    Returns:
        None
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

    Args:
        y (list | None): Optional {var} data for comparison (e.g., ERA5). 
                            If None, only `x` and `z` are plotted.
        x (list): {var} data for biased dataset.
        z (list): {var} data for the corrected dataset.
        dates (list): List of dates corresponding to the {var} data.
        title (str): Title for the plot.
        savedir (str): File path to save the generated plot.

    Returns:
        None: The function saves the plot to the specified directory.
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

    Args:
        y (Optional[np.ndarray]): An optional array of values. If None, it will be set to `x`.
        x (np.ndarray): An array of values to calculate the monthly mean for.
        z (np.ndarray): Another array of values to calculate the monthly mean for.
        dates (List[str]): A list of date strings corresponding to the input arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
    """
    if y is None:
        y = x
    df = pd.DataFrame({'dates' : dates,
                                'y': y,
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
    
    parser = argparse.ArgumentParser(description="Predict and plot results")
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)') 
    parser.add_argument('--ssp', type=str, help='SSP scenario (e.g., ssp585)')
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
    dataset = CONFIG[exp]['dataset']
    

    get_data_bc = Data([-12.5, 27.5, 31., 71.]) # Europeen domain
    gcm_ds = get_data_bc.get_gcm_dataset(var, None)
    lon, lat = gcm_ds.lon.values, gcm_ds.lat.values
    get_data = Data(domain=domain)

    debiaser = CDFt.from_variable(variable=var,
                                  apply_by_month=True)

    train_hist = dict(np.load(DATASET_BC_DIR/f'bc_train_hist_{simu}.npz', allow_pickle=True))
    test_hist = dict(np.load(DATASET_BC_DIR/f'bc_test_hist_{simu}.npz', allow_pickle=True))
    test_future = dict(np.load(DATASET_BC_DIR/f'bc_test_future_{simu}.npz', allow_pickle=True))

   
    ##### 1980-1999
    train_hist_bc = debiaser.apply(obs=train_hist['era5'],
                                cm_hist=train_hist[simu], 
                                cm_future=train_hist[simu], 
                                time_obs=train_hist['dates'], 
                                time_cm_hist=train_hist['dates'],
                                time_cm_future=train_hist['dates'])
    ##### 2000-2014
    test_hist_bc = debiaser.apply(obs=train_hist['era5'],
                                cm_hist=train_hist[simu], 
                                cm_future=test_hist[simu], 
                                time_obs=train_hist['dates'], 
                                time_cm_hist=train_hist['dates'],
                                time_cm_future=test_hist['dates'])
    
    ##### 2015-2100
    test_future_bc = debiaser.apply(obs=train_hist['era5'],
                            cm_hist=train_hist[simu], 
                            cm_future=test_future[simu], 
                            time_obs=train_hist['dates'], 
                            time_cm_hist=train_hist['dates'],
                            time_cm_future=test_future['dates'])
    

    var_marginal_bias_data = marginal.calculate_marginal_bias(metrics = [metrics.cold_days, metrics.warm_days], 
                                                            percentage_or_absolute='absolute',
                                                            obs = test_hist['era5'],
                                                            raw = test_hist[simu],
                                                            CDFt = test_hist_bc)
    plot = marginal.plot_marginal_bias(variable = var,
                                       bias_df = var_marginal_bias_data)
    plot.savefig(GRAPHS_DIR /f'biascorrection/{var}_ibicus_bias_boxplot_{simu}.png')
    
    var_trend_bias_data = trend.calculate_future_trend_bias(statistics = ["mean", 0.05, 0.95], 
                                                        trend_type = 'additive',
                                                        raw_validate = test_hist[simu], raw_future = test_future[simu],
                                                        metrics = [metrics.cold_days, metrics.warm_days],
                                                        CDFt = [test_hist_bc, test_future_bc])

    plot = trend.plot_future_trend_bias_boxplot(variable =var, bias_df = var_trend_bias_data,remove_outliers = True)
    plot.savefig(GRAPHS_DIR / f'biascorrection/{var}_ibicus_bias_futur_trend_{ssp}_{simu}.png')

    Y0 = np.mean(train_hist['era5'], axis=(1,2))
    X0 = np.mean(train_hist[simu], axis=(1,2))
    Z0 = np.mean(train_hist_bc, axis=(1,2))
    Y1 = np.mean(test_hist['era5'], axis=(1,2))
    X1 = np.mean(test_hist[simu], axis=(1,2))
    Z1 = np.mean(test_hist_bc, axis=(1,2))
    X2 = np.mean(test_future[simu], axis=(1,2))
    Z2 = np.mean(test_future_bc, axis=(1,2))

    print(Y0.shape, X0.shape, Z0.shape)
    print(Y1.shape, X1.shape, Z1.shape)
    print(X2.shape, Z2.shape)

    ########## TEMPORAL PROFILES
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


    
  

    plt.figure(figsize=(6, 6))
    plt.scatter(X1, Z1, color='blue', s=5, label='2000-2014')
    plt.scatter(X2, Z2, color='green', s=5, label = f'2015-2100 {ssp}')
    plt.plot(np.arange(270,300), np.arange(270,300), color='black')
    plt.xlim(270,300)
    plt.ylim(270,300)
    plt.legend()
    plt.xlabel(f'{var} {simu} (K)')
    plt.ylabel(f'{var} {simu} bc (K)')
    plt.title(f'Daily mean {var} over the historical and future test period')
    plt.savefig(GRAPHS_DIR /f'biascorrection/{var}_ibicus_test_hist_linear_{ssp}_{simu}.png')


    df_simu = pd.DataFrame({'dates' : np.concatenate((train_hist['dates'], 
                                             test_hist['dates'], 
                                             test_future['dates']), axis=None),
                   'values' : np.concatenate((X0, 
                                              X1, 
                                              X2), axis=None),
                   'labels' : np.concatenate((np.ones_like(X0),
                                              2*np.ones_like(X1),
                                              3*np.ones_like(X2)), axis=None)})
    df_simu['year'] = pd.to_datetime(df_simu['dates']).dt.year
    df_simu_year = df_simu.groupby('year').mean()

    df_era5 = pd.DataFrame({'dates' : np.concatenate((train_hist['dates'], 
                                                test_hist['dates']), axis=None),
                    'values' : np.concatenate((Y0, 
                                                Y1), axis=None),
                    'labels' : np.concatenate((np.ones_like(Y0),
                                                2*np.ones_like(Y1)), axis=None)})
    df_era5['year'] = pd.to_datetime(df_era5['dates']).dt.year
    df_era5_year = df_era5.groupby('year').mean()

    df_simu_bc = pd.DataFrame({'dates' : np.concatenate((train_hist['dates'], 
                                             test_hist['dates'], 
                                             test_future['dates']), axis=None),
                   'values' : np.concatenate((Z0, 
                                              Z1, 
                                              Z2), axis=None),
                   'labels' : np.concatenate((np.ones_like(Z0),
                                              2*np.ones_like(Z1),
                                              3*np.ones_like(Z2)), axis=None)})
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
    plt.ylabel('{var} (K)')
    plt.legend()
    plt.savefig(GRAPHS_DIR/f'biascorrection/{var}_bc_datasets_temporal_profiles_ibicus_{ssp}_{simu}.png')

    ######### HISTOGRAM PROFILES
    Y0, X0, Z0, dates = monthly_mean(Y0, X0, Z0, train_hist['dates'])
    plot_seasonal_hist(Y0, X0, Z0, dates,
                       title =f'Monthly mean {var} over the historical Train period (1980-1999)',
                       savedir=GRAPHS_DIR/f'biascorrection/{var}_ibicus_train_hist_histo_{simu}.png',
                       simu=simu)
    Y1, X1, Z1, dates = monthly_mean(Y1, X1, Z1, test_hist['dates'])
    plot_seasonal_hist(Y1, X1, Z1, dates,
                       title =f'Monthly mean {var} over the historical Test period (2000-2014)',
                       savedir=GRAPHS_DIR /f'biascorrection/{var}_ibicus_test_hist_histo_{simu}.png',
                        simu=simu)
    _, X2, Z2, dates = monthly_mean(None, X2, Z2, test_future['dates'])
    plot_seasonal_hist(None, X2, Z2, dates,
                       title =f'Monthly mean {var} over the future Test period (2015-2100 {ssp})',
                       savedir=GRAPHS_DIR/f'biascorrection/{var}_ibicus_test_future_histo_{ssp}_{simu}.png',
                       simu=simu)

    ds_train_hist_bc = xr.Dataset(data_vars=dict(
                            tas=(['time', 'lat', 'lon'], train_hist_bc)),
                            coords=dict(
                            lat=('lat', lat),
                            lon=('lon', lon),
                            time=('time', train_hist['dates'])
                            ))
    ds_test_hist_bc = xr.Dataset(data_vars=dict(
                            tas=(['time', 'lat', 'lon'], test_hist_bc)),
                            coords=dict(
                            lat=('lat', lat),
                            lon=('lon', lon),
                            time=('time', test_hist['dates'])
                            ))
    
    ds_test_future_bc = xr.Dataset(data_vars=dict(
                            tas=(['time', 'lat', 'lon'], test_future_bc)),
                            coords=dict(
                            lat=('lat', lat),
                            lon=('lon', lon),
                            time=('time', test_future['dates'])
                            ))
    if simu == 'gcm':
        ds_train_hist_bc.to_netcdf(GCM_RAW_DIR/f'CNRM-CM6-1-BC/{var}_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc')
        ds_test_hist_bc.to_netcdf(GCM_RAW_DIR/f'CNRM-CM6-1-BC/{var}_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc')
        ds_test_future_bc.to_netcdf(GCM_RAW_DIR/f'CNRM-CM6-1-BC/{var}_day_CNRM-CM6-1_{ssp}_r1i1p1f2_gr_20150101-21001231_bc.nc')
    elif simu == 'rcm':
        ds_train_hist_bc.to_netcdf(RCM_RAW_DIR/f'ALADIN-BC/{var}_day_ALADIN_historical_r1i1p1f2_gr_19800101-19991231_150km_bc.nc')
        ds_test_hist_bc.to_netcdf(RCM_RAW_DIR/f'ALADIN-BC/{var}_day_ALADIN_historical_r1i1p1f2_gr_20000101-20141231_150km_bc.nc')
        ds_test_future_bc.to_netcdf(RCM_RAW_DIR/f'ALADIN-BC/{var}_day_ALADIN_{ssp}_r1i1p1f2_gr_20150101-21001231_150km_bc.nc')
    
    for date in DATES_BC_TRAIN_HIST:
        print(date)
        x = []

        ds = xr.open_dataset(orog_file)
        x.append(ds['elevation'].values)
        ds_train_hist_bc_i = ds_train_hist_bc.sel(time=ds_train_hist_bc.time.dt.date == date.date())
        ds_train_hist_bc_i = ds_train_hist_bc_i.isel(time=0, drop=True)

        ds_train_hist_bc_i = reformat_as_target(ds_train_hist_bc_i, 
                                         target_file=target_file,
                                         domain=domain, 
                                         method="conservative_normed",
                                         mask=True
                                         )
        

    
        x.append(ds_train_hist_bc_i.tas.values)
        x = np.stack(x, axis = 0)
        ds_target = get_data.get_target_dataset(target=CONFIG[exp]['target'], 
                                     var = var, 
                                     date = date)
        y = ds_target[var].values
        y = np.expand_dims(y, axis= 0)
        
        sample = {'x' : x,
                    'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_BC_DIR/f'dataset_{exp}_test_{simu}_bc/sample_{date_str}.npz', **sample)


    
    for date in DATES_BC_TEST_HIST:
        print(date)
        x = []

        ds = xr.open_dataset(orog_file)
        x.append(ds['elevation'].values)

        ds_test_hist_bc_i = ds_test_hist_bc.sel(time=ds_test_hist_bc.time.dt.date == date.date())
        ds_test_hist_bc_i = ds_test_hist_bc_i.isel(time=0, drop=True)

        ds_test_hist_bc_i = reformat_as_target(ds_test_hist_bc_i, 
                                         target_file=target_file,
                                         domain=domain, 
                                         method="conservative_normed",
                                         mask=True
                                         )
                                         
    
        x.append(ds_test_hist_bc_i.tas.values)
        x = np.stack(x, axis = 0)
        ds_target = get_data.get_target_dataset(target=CONFIG[exp]['target'], 
                                     var = var, 
                                     date = date)
        y = ds_target[var].values
        y = np.expand_dims(y, axis= 0)
        
        sample = {'x' : x,
                    'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_BC_DIR/f'dataset_{exp}_test_{simu}_bc/sample_{date_str}.npz', **sample)

    

    for date in DATES_BC_TEST_FUTURE:
        print(date)
        x = []

        ds = xr.open_dataset(orog_file)
        x.append(ds['elevation'].values)

        ds_test_future_bc_i = ds_test_future_bc.sel(time=ds_test_future_bc.time.dt.date == date.date())
        ds_test_future_bc_i = ds_test_future_bc_i.isel(time=0, drop=True)
        ds_test_future_bc_i = reformat_as_target(ds_test_future_bc_i, 
                                         target_file=target_file,
                                         domain=domain, 
                                         method="conservative_normed",
                                         mask=True
                                         )  
    
        x.append(ds_test_future_bc_i.tas.values)
        x = np.stack(x, axis = 0)
        
        sample = {'x' : x}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_BC_DIR/f'dataset_{exp}_test_{simu}_bc/sample_{date_str}.npz', **sample)
    