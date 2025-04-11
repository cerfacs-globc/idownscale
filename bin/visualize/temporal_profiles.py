import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import xarray as xr
import torch
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from iriscc.settings import DATASET_BC_DIR


train_hist = dict(np.load(DATASET_BC_DIR/'bc_train_hist.npz'))
test_hist = dict(np.load(DATASET_BC_DIR/'bc_test_hist.npz'))
test_future = dict(np.load(DATASET_BC_DIR/'bc_test_future.npz'))

era5_hist = np.mean(train_hist['era5'], axis = (1,2))
cmip6_hist = np.mean(train_hist['cmip6'], axis = (1,2))
dates_hist = train_hist['dates']

era5_test = np.mean(test_hist['era5'], axis = (1,2))
cmip6_test = np.mean(test_hist['cmip6'], axis = (1,2))
dates_test = test_hist['dates']

cmip6_test_future = np.mean(test_future['cmip6'], axis = (1,2))
dates_test_future = test_future['dates']

df_cmip6 = pd.DataFrame({'dates' : np.concatenate((dates_hist, 
                                             dates_test, 
                                             dates_test_future), axis=None),
                   'values' : np.concatenate((cmip6_hist, 
                                              cmip6_test, 
                                              cmip6_test_future), axis=None),
                   'labels' : np.concatenate((np.ones_like(cmip6_hist),
                                              2*np.ones_like(cmip6_test),
                                              3*np.ones_like(cmip6_test_future)), axis=None)})
df_cmip6['year'] = pd.to_datetime(df_cmip6['dates']).dt.year
df_cmip6_year = df_cmip6.groupby('year').mean()

df_era5 = pd.DataFrame({'dates' : np.concatenate((dates_hist, 
                                             dates_test), axis=None),
                   'values' : np.concatenate((era5_hist, 
                                              era5_test), axis=None),
                   'labels' : np.concatenate((np.ones_like(era5_hist),
                                              2*np.ones_like(era5_test)), axis=None)})
df_era5['year'] = pd.to_datetime(df_era5['dates']).dt.year
df_era5_year = df_era5.groupby('year').mean()

plt.figure(figsize=(8, 4))
plt.plot(df_era5_year.index, np.where(df_era5_year["labels"]==1., df_era5_year["values"], None), color="red", label="ERA5")
plt.plot(df_cmip6_year.index, np.where(df_cmip6_year["labels"]==1., df_cmip6_year["values"], None), color="blue", label="CNRM-CM6-1")
plt.plot(df_era5_year.index, np.where(df_era5_year["labels"]==2., df_era5_year["values"], None), color="red")
plt.plot(df_cmip6_year.index, np.where(df_cmip6_year["labels"]==2., df_cmip6_year["values"], None), color="blue")
plt.plot(df_cmip6_year.index, np.where(df_cmip6_year["labels"]==2., df_cmip6_year["values"] - 2, None), color="green", label="CNRM-CM6-1 bc")
plt.plot(df_cmip6_year.index, np.where(df_cmip6_year["labels"]==3., df_cmip6_year["values"], None), color="blue")
plt.plot(df_cmip6_year.index, np.where(df_cmip6_year["labels"]==3., df_cmip6_year["values"] - 2, None), color="green")

plt.title('Bias correction Train and Test Datasets')
plt.ylabel('Temperature (K)')
plt.legend()
plt.savefig('/gpfs-calypso/scratch/globc/garcia/graph/test/bc_datasets_temporal_profiles.png')

