''' Experience 2 : CMIP6 BC and topography as input and SAFRAN as target'''

import sys
sys.path.append('.')

import xarray as xr
import numpy as np
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from ibicus.evaluate import marginal, metrics, trend
from ibicus.debias import CDFt, LinearScaling, ISIMIP

from iriscc.plotutils import plot_test
from iriscc.datautils import reformat_as_target, standardize_dims_and_coords, standardize_longitudes, remove_countries
from iriscc.settings import (SAFRAN_REFORMAT_DIR, 
                             GRAPHS_DIR,
                             DATASET_BC_CMIP6_ERA5,
                             ERA5_DIR,
                             DATES_TEST,
                             CONFIG,
                             OROG_FILE,
                             TARGET_SAFRAN_FILE,
                             DATES_BC_TEST_FUTURE,
                             DATES_BC_TEST_HIST,
                             DATASET_BC_DIR,
                             TARGET)

def target_data(date):
    ''' Returns target data as an array of shape (H, W) '''

    threshold_date = datetime(date.year, 8, 1)
    year = date.year
    if date < threshold_date : 
        year = year-1
    ds = xr.open_dataset(glob.glob(str(SAFRAN_REFORMAT_DIR/f"SAFRAN_{year}080107_{year+1}080106_reformat.nc"))[0])

    if date == datetime(date.year, 8, 1):
        ds_before = xr.open_dataset(glob.glob(str(SAFRAN_REFORMAT_DIR/f"SAFRAN_{year-1}080107_{year}080106_reformat.nc"))[0])
        ds_before = ds_before.isel(time=slice (-7, None))
        ds = ds.isel(time = slice(None, 17))
        ds = ds.merge(ds_before)
    else : 
        ds = ds.sel(time=pd.date_range(start=date.strftime("%Y-%m-%d"), periods = 24, freq='h').to_numpy())
    y = ds[TARGET].values.mean(axis=0)
    y = remove_countries(y)
    y = np.expand_dims(y, axis=0)
    return y

def plot_tprofiles_short_range(y, x, z, title, savedir):
    plt.figure(figsize=(15, 5))
    if y is not None:
        plt.plot(y[1000:2000],label='ERA5', color='red')
    plt.plot(x[1000:2000],label='CNRM-CM6-1', color='blue')
    plt.plot(z[1000:2000],label='CNRM-CM6-1 bc', color='green')
    plt.xlabel('Days')
    plt.ylabel('Daily temperature ')
    plt.title(title)
    plt.legend()
    plt.savefig(savedir)

def plot_seasonal_hist(y, x, z, dates, title, savedir):
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
    ax1.hist([x[i] for i in i_winter], histtype='step', color='blue', label='CNRM-CM6-1', density=True, range=(265,300))
    ax1.hist([z[i] for i in i_winter], histtype='step', color='green', label='CNRM-CM6-1 bc', density=True)
    ax2.hist([x[i] for i in i_summer], histtype='step', color='blue', label='CNRM-CM6-1', density=True, range=(265,300))
    ax2.hist([z[i] for i in i_summer], histtype='step', color='green', label='CNRM-CM6-1 bc', density=True)
    plt.suptitle(title)
    ax1.set_title('Winter')
    ax2.set_title('Summer')
    ax1.set_xlabel('Temperature (K)')
    ax2.set_xlabel('Temperature (K)')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(savedir)
    
def monthly_mean(y, x, z, dates):
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
    
    #debiaser = CDFt.from_variable(variable="tas",
    #                              apply_by_month=True)
    #debiaser = CDFt.from_variable(variable = 'tas', 
    #                   running_window_length=91, 
    #                   running_window_step_length=31, 
    #                   running_window_over_years_of_cm_future_length=31, 
    #                   running_window_over_years_of_cm_future_step_length=10)

    debiaser = ISIMIP.from_variable('tas')
    ssp = 'ssp585'

    train_hist = dict(np.load(DATASET_BC_DIR/'bc_train_hist.npz', allow_pickle=True))
    test_hist = dict(np.load(DATASET_BC_DIR/'bc_test_hist.npz', allow_pickle=True))
    test_future = dict(np.load(DATASET_BC_DIR/'bc_test_future.npz', allow_pickle=True))
    coordinates = dict(np.load(DATASET_BC_DIR / 'coordinates.npz', allow_pickle=True))

   
    ##### 1980-1999
    train_hist_bc = debiaser.apply(obs=train_hist['era5'],
                                cm_hist=train_hist['cmip6'], 
                                cm_future=train_hist['cmip6'], 
                                time_obs=train_hist['dates'], 
                                time_cm_hist=train_hist['dates'],
                                time_cm_future=train_hist['dates'])
    ##### 2000-2014
    test_hist_bc = debiaser.apply(obs=train_hist['era5'],
                                cm_hist=train_hist['cmip6'], 
                                cm_future=test_hist['cmip6'], 
                                time_obs=train_hist['dates'], 
                                time_cm_hist=train_hist['dates'],
                                time_cm_future=test_hist['dates'])
    
    ##### 2015-2100
    test_future_bc = debiaser.apply(obs=train_hist['era5'],
                            cm_hist=train_hist['cmip6'], 
                            cm_future=test_future['cmip6'], 
                            time_obs=train_hist['dates'], 
                            time_cm_hist=train_hist['dates'],
                            time_cm_future=test_future['dates'])
    

    tas_marginal_bias_data = marginal.calculate_marginal_bias(metrics = [metrics.cold_days, metrics.warm_days], 
                                                            percentage_or_absolute='absolute',
                                                            obs = test_hist['era5'],
                                                            raw = test_hist['cmip6'],
                                                            CDFt = test_hist_bc)
    print(tas_marginal_bias_data)
    plot = marginal.plot_marginal_bias(variable = 'tas',
                                       bias_df = tas_marginal_bias_data)
    plot.savefig(GRAPHS_DIR /'test/ibicus_bias_boxplot.png')
    
    tas_trend_bias_data = trend.calculate_future_trend_bias(statistics = ["mean", 0.05, 0.95], 
                                                        trend_type = 'additive',
                                                        raw_validate = test_hist['cmip6'], raw_future = test_future['cmip6'],
                                                        metrics = [metrics.cold_days, metrics.warm_days],
                                                        CDFt = [test_hist_bc, test_future_bc])

    plot = trend.plot_future_trend_bias_boxplot(variable ='tas', bias_df = tas_trend_bias_data,remove_outliers = True)
    plot.savefig(GRAPHS_DIR / f'test/ibicus_bias_futur_trend_{ssp}.png')

    Y0 = np.mean(train_hist['era5'], axis=(1,2))
    X0 = np.mean(train_hist['cmip6'], axis=(1,2))
    Z0 = np.mean(train_hist_bc, axis=(1,2))
    Y1 = np.mean(test_hist['era5'], axis=(1,2))
    X1 = np.mean(test_hist['cmip6'], axis=(1,2))
    Z1 = np.mean(test_hist_bc, axis=(1,2))
    X2 = np.mean(test_future['cmip6'], axis=(1,2))
    Z2 = np.mean(test_future_bc, axis=(1,2))

    print(Y0.shape, X0.shape, Z0.shape)
    print(Y1.shape, X1.shape, Z1.shape)
    print(X2.shape, Z2.shape)

    ########## TEMPORAL PROFILES
    plot_tprofiles_short_range(Y0, X0, Z0, 
                               title = 'Daily temperature over the historical Train period (1980-1999)',
                               savedir=GRAPHS_DIR / 'test/ibicus_train_hist_tprofiles.png')
    plot_tprofiles_short_range(Y1, X1, Z1, 
                               title = 'Daily temperature over the historical Test period (2000-2014)',
                               savedir=GRAPHS_DIR / 'test/ibicus_test_hist_tprofiles.png')
    plot_tprofiles_short_range(None, X2, Z2, 
                               title = f'Daily temperature over the future Test period (2015-2100 {ssp})',
                               savedir=GRAPHS_DIR / f'test/ibicus_test_future_tprofiles_{ssp}.png')


    
  

    plt.figure(figsize=(6, 6))
    plt.scatter(X1, Z1, color='blue', s=5, label='2000-2014')
    plt.scatter(X2, Z2, color='green', s=5, label = f'2015-2100 {ssp}')
    plt.plot(np.arange(270,300), np.arange(270,300), color='black')
    plt.xlim(270,300)
    plt.ylim(270,300)
    plt.legend()
    plt.xlabel('Temperature CNRM-CM6-1 (K)')
    plt.ylabel('Temperature CNRM-CM6-1 bc (K)')
    plt.title('Daily mean temperature over the historical and future test period')
    plt.savefig(GRAPHS_DIR /f'test/ibicus_test_hist_linear_{ssp}.png')


    df_cmip6 = pd.DataFrame({'dates' : np.concatenate((train_hist['dates'], 
                                             test_hist['dates'], 
                                             test_future['dates']), axis=None),
                   'values' : np.concatenate((X0, 
                                              X1, 
                                              X2), axis=None),
                   'labels' : np.concatenate((np.ones_like(X0),
                                              2*np.ones_like(X1),
                                              3*np.ones_like(X2)), axis=None)})
    df_cmip6['year'] = pd.to_datetime(df_cmip6['dates']).dt.year
    df_cmip6_year = df_cmip6.groupby('year').mean()

    df_era5 = pd.DataFrame({'dates' : np.concatenate((train_hist['dates'], 
                                                test_hist['dates']), axis=None),
                    'values' : np.concatenate((Y0, 
                                                Y1), axis=None),
                    'labels' : np.concatenate((np.ones_like(Y0),
                                                2*np.ones_like(Y1)), axis=None)})
    df_era5['year'] = pd.to_datetime(df_era5['dates']).dt.year
    df_era5_year = df_era5.groupby('year').mean()

    df_cmip6_bc = pd.DataFrame({'dates' : np.concatenate((train_hist['dates'], 
                                             test_hist['dates'], 
                                             test_future['dates']), axis=None),
                   'values' : np.concatenate((Z0, 
                                              Z1, 
                                              Z2), axis=None),
                   'labels' : np.concatenate((np.ones_like(Z0),
                                              2*np.ones_like(Z1),
                                              3*np.ones_like(Z2)), axis=None)})
    df_cmip6_bc['year'] = pd.to_datetime(df_cmip6_bc['dates']).dt.year
    df_cmip6_bc_year = df_cmip6_bc.groupby('year').mean()

    plt.figure(figsize=(8, 4))
    plt.plot(df_era5_year.index, np.where(df_era5_year["labels"]==1., df_era5_year["values"], None), color="red", label='ERA5')
    plt.plot(df_cmip6_year.index, np.where(df_cmip6_year["labels"]==1., df_cmip6_year["values"], None), color="blue", label='CNRM-CM6-1')
    plt.plot(df_cmip6_bc_year.index, np.where(df_cmip6_bc_year["labels"]==1., df_cmip6_bc_year["values"], None), color="green", label='CNRM-CM6-1 bc')
    plt.plot(df_era5_year.index, np.where(df_era5_year["labels"]==2., df_era5_year["values"], None), color="red")
    plt.plot(df_cmip6_year.index, np.where(df_cmip6_year["labels"]==2., df_cmip6_year["values"], None), color="blue")
    plt.plot(df_cmip6_bc_year.index, np.where(df_cmip6_bc_year["labels"]==2., df_cmip6_bc_year["values"], None), color="green")
    plt.plot(df_cmip6_year.index, np.where(df_cmip6_year["labels"]==3., df_cmip6_year["values"], None), color="blue")
    plt.plot(df_cmip6_bc_year.index, np.where(df_cmip6_bc_year["labels"]==3., df_cmip6_bc_year["values"], None), color="green")
    plt.title(f'Annual mean temperature ({ssp})')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.savefig(f'/gpfs-calypso/scratch/globc/garcia/graph/test/bc_datasets_temporal_profiles_ibicus_{ssp}.png')

    ######### HISTOGRAM PROFILES
    Y0, X0, Z0, dates = monthly_mean(Y0, X0, Z0, train_hist['dates'])
    plot_seasonal_hist(Y0, X0, Z0, dates,
                       title ='Monthly mean temperature over the historical Train period (1980-1999)',
                       savedir=GRAPHS_DIR/'test/ibicus_train_hist_histo.png')
    Y1, X1, Z1, dates = monthly_mean(Y1, X1, Z1, test_hist['dates'])
    plot_seasonal_hist(Y1, X1, Z1, dates,
                       title ='Monthly mean temperature over the historical Test period (2000-2014)',
                       savedir=GRAPHS_DIR /'test/ibicus_test_hist_histo.png')
    _, X2, Z2, dates = monthly_mean(None, X2, Z2, test_future['dates'])
    plot_seasonal_hist(None, X2, Z2, dates,
                       title =f'Monthly mean temperature over the future Test period (2015-2100 {ssp})',
                       savedir=GRAPHS_DIR/f'test/ibicus_test_future_histo_{ssp}.png')

    '''                   
    ds_test_hist_bc = xr.Dataset(data_vars=dict(
                            tas=(['time', 'lat', 'lon'], test_hist_bc)),
                            coords=dict(
                            lat=('lat', coordinates['lat']),
                            lon=('lon', coordinates['lon']),
                            time=('time', test_hist['dates'])
                            ))

    ds_test_future_bc = xr.Dataset(data_vars=dict(
                            tas=(['time', 'lat', 'lon'], test_future_bc)),
                            coords=dict(
                            lat=('lat', coordinates['lat']),
                            lon=('lon', coordinates['lon']),
                            time=('time', test_future['dates'])
                            ))
    
    for date in DATES_BC_TEST_HIST:
        print(date)
        x = []

        # Commune variables
        ds = xr.open_dataset(OROG_FILE)
        x.append(ds['Altitude'].values)
        ### run en mode interactif et mettre des print partout
        ds_test_hist_bc_i = ds_test_hist_bc.sel(time=ds_test_hist_bc.time.dt.date == date.date())
        ds_test_hist_bc_i = ds_test_hist_bc_i.isel(time=0, drop=True)

        ds_test_hist_bc_i = reformat_as_target(ds_test_hist_bc_i, 
                                         target_file=TARGET_SAFRAN_FILE,
                                         domain=CONFIG['safran']['domain']['france'], 
                                         method="conservative_normed")
    
        x.append(ds_test_hist_bc_i.tas.values)
        x = np.stack(x, axis = 0)
        y = target_data(date)
        
        sample = {'x' : x,
                    'y' : y}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_BC_DIR/f'dataset_exp3_test_cmip6_bc/sample_{date_str}.npz', **sample)

    for date in DATES_BC_TEST_FUTURE:
        print(date)
        x = []

        # Commune variables
        ds = xr.open_dataset(OROG_FILE)
        x.append(ds['Altitude'].values)

        ds_test_future_bc_i = ds_test_future_bc.sel(time=ds_test_future_bc.time.dt.date == date.date())
        ds_test_future_bc_i = ds_test_future_bc_i.isel(time=0, drop=True)
        ds_test_future_bc_i = reformat_as_target(ds_test_future_bc_i, 
                                         target_file=TARGET_SAFRAN_FILE,
                                         domain=CONFIG['safran']['domain']['france'], 
                                         method="conservative_normed")
    
        x.append(ds_test_future_bc_i.tas.values)
        x = np.stack(x, axis = 0)
        
        sample = {'x' : x}
        date_str = date.date().strftime('%Y%m%d')
        np.savez(DATASET_BC_DIR/f'dataset_exp3_test_cmip6_bc/sample_{date_str}.npz', **sample)
        '''