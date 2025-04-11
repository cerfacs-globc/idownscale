import sys
sys.path.append('.')

import glob
import numpy as np
import xarray as xr
from ibicus.debias import CDFt
import matplotlib.pyplot as plt


from iriscc.settings import DATASET_TEST_BC_CMIP6_DIR, DATASET_BC_CMIP6_ERA5, DATES_TEST, DATASET_EXP2_DIR, DATASET_EXP1_DIR
from iriscc.plotutils import plot_test

mean_era5 = []
mean_cmip6 = []
mean_cmip6bc = []

dates = []

for date in DATES_TEST:
    print(date)
    date_str = date.strftime('%Y%m%d')  # Conversion en format string
    dates.append(date)
    
    # Chargement des données ERA5
    sample = glob.glob(str(DATASET_EXP2_DIR/f'sample_{date_str}.npz'))[0]
    data_era5 = np.load(sample, allow_pickle=True)
    era5 = data_era5['x'][1]
    mean_era5.append(np.mean(era5))
    
    # Chargement des données CMIP6
    sample = glob.glob(str(DATASET_EXP1_DIR/f'sample_{date_str}.npz'))[0]
    data_cmip6 = np.load(sample, allow_pickle=True)
    cmip6 = data_cmip6['x'][1]
    mean_cmip6.append(np.mean(cmip6))
    
    # Chargement des données CMIP6BC
    sample = glob.glob(str(DATASET_TEST_BC_CMIP6_DIR/f'sample_{date_str}.npz'))[0]
    data_cmip6bc = np.load(sample, allow_pickle=True)
    cmip6bc = data_cmip6bc['x'][1]
    mean_cmip6bc.append(np.mean(cmip6bc))

# Calcul des erreurs
mean_cmip6_error = np.array(mean_cmip6) - np.array(mean_era5)
mean_cmip6bc_error = np.array(mean_cmip6bc) - np.array(mean_era5)

rmse_cmip6 = np.sqrt(np.mean(mean_cmip6_error**2))
rmse_cmip6bc = np.sqrt(np.mean(mean_cmip6bc_error**2))

bias_cmip6 = np.mean(mean_cmip6_error)
bias_cmip6bc = np.mean(mean_cmip6bc_error)

# Tracé des courbes
plt.figure(figsize=(12, 6))
plt.plot(dates, mean_era5, label='ERA5', color='blue')
plt.plot(dates, mean_cmip6, label='CMIP6', color='red')
plt.plot(dates, mean_cmip6bc, label='CMIP6BC', color='green')
plt.xlabel('Temps')
plt.ylabel('Température Moyenne')
plt.title('Évolution des températures moyennes')
plt.legend()
plt.grid()
plt.savefig('/scratch/globc/garcia/graph/test.png')

# Tracé du scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(mean_era5, mean_cmip6, label='CMIP6 vs ERA5', color='red')
plt.scatter(mean_era5, mean_cmip6bc, label='CMIP6BC vs ERA5', color='green')
plt.plot(np.arange(270,300), np.arange(270,300), color='black')
plt.xlim(270,300)
plt.ylim(270,300)
plt.xlabel('Température Moyenne ERA5')
plt.ylabel('Température Moyenne CMIP6 / CMIP6BC')
plt.title('Comparaison des températures moyennes')
plt.legend()
plt.grid()
plt.savefig('/scratch/globc/garcia/graph/test2.png')

# Affichage des erreurs
print(f"RMSE CMIP6 vs ERA5: {rmse_cmip6}")
print(f"RMSE CMIP6BC vs ERA5: {rmse_cmip6bc}")
print(f"Biais CMIP6 vs ERA5: {bias_cmip6}")
print(f"Biais CMIP6BC vs ERA5: {bias_cmip6bc}")

#plot_test(xinit[1], 'xinit', '/scratch/globc/garcia/graph/test.png')
#plot_test(x, 'x 20040101', '/scratch/globc/garcia/graph/test.png')
#plot_test(yinit[0], 'yinit', '/scratch/globc/garcia/graph/test3.png')
#plot_test(y, 'y 20040101', '/scratch/globc/garcia/graph/test4.png')
#plot_test(y_hat, 'yhat', '/scratch/globc/garcia/graph/test.png')
