""" 
Bias correction using SBCK python library. This script is just a test.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import SBCK as bc
import numpy as np
import matplotlib.pyplot as plt
from iriscc.settings import DATASET_BC_DIR, GRAPHS_DIR




train_hist = dict(np.load(DATASET_BC_DIR/'bc_train_hist.npz'))
test_hist = dict(np.load(DATASET_BC_DIR/'bc_test_hist.npz'))
test_future = dict(np.load(DATASET_BC_DIR/'bc_test_future.npz'))


Y0 = train_hist['era5'][:5479]
X0 = train_hist['gcm'][:5479]
Y1 = test_hist['era5']  # used for test 5479
X1 = test_hist['gcm']
X2 = test_future['gcm'][-5479:]

t, lat, lon = Y0.shape

Y0 = Y0.reshape(Y0.shape[0], -1)
X0 = X0.reshape(X0.shape[0], -1)
X1 = X1.reshape(X1.shape[0], -1)
X2 = X2.reshape(X2.shape[0], -1)
#Y0 = np.expand_dims(Y0.flatten(), axis=1)
#X0 = np.expand_dims(X0.flatten(), axis=1)
#X1 = np.expand_dims(X1.flatten(), axis=1)


############ HISTORICAL DEBIAS 
cdft = bc.CDFt(version=2, normalize_cdf=True)
Z0 = np.zeros(Y0.shape)
Z1 = np.zeros(Y0.shape)
Z2 = np.zeros(Y0.shape)

for cell in range(lat*lon):
    cdft.fit(Y0[:,cell],X0[:,cell],X1[:,cell])

    Z1_cell, Z0_cell = cdft.predict( X1[:,cell] , X0[:,cell] )
    Z1[:,cell], Z0[:,cell] = Z1_cell[:,0], Z0_cell[:,0]

    Z2_cell, Z0_cell = cdft.predict( X2[:,cell] , X0[:,cell] )
    Z2[:,cell], Z0[:,cell] = Z2_cell[:,0], Z0_cell[:,0]
print(Z1.shape)


########## TEMPORAL PROFILES
plt.figure(figsize=(20, 8))
plt.plot(np.mean(Y0, axis=1)[1000:2000],label='ERA5', color='red')
plt.plot(np.mean(X0, axis=1)[1000:2000],label='CNRM-CM6-1', color='blue')
plt.plot(np.mean(Z0, axis=1)[1000:2000],label='CNRM-CM6-1 bc', color='green')
plt.title('Historical Train period (1980-1994) SBCK')
plt.legend()
plt.savefig(GRAPHS_DIR / 'test/exp3_sbck_train_hist_tprofiles.png')

plt.figure(figsize=(20, 8))
plt.plot(np.mean(Y1, axis=(1,2))[1000:2000],label='ERA5', color='red')
plt.plot(np.mean(X1, axis=1)[1000:2000],label='CNRM-CM6-1', color='blue')
plt.plot(np.mean(Z1, axis=1)[1000:2000],label='CNRM-CM6-1 bc', color='green')
plt.title('Historical Test period (2000-2014) SBCK')
plt.legend()
plt.savefig(GRAPHS_DIR / 'test/exp3_sbck_test_hist_tprofiles.png')

plt.figure(figsize=(20, 8))
plt.plot(np.mean(X2, axis=1)[1000:2000],label='CNRM-CM6-1', color='blue')
plt.plot(np.mean(Z2, axis=1)[1000:2000],label='CNRM-CM6-1 bc', color='green')
plt.title('Future Test period (2015-2029) SBCK')
plt.legend()
plt.savefig(GRAPHS_DIR / 'test/exp3_sbck_test_future_tprofiles.png')


######### HISTOGRAM PROFILES
print(Y0.shape, X0.shape, Z0.shape)
plt.figure(figsize=(6,6))
plt.hist(np.mean(Y0, axis=1), histtype='step', color='red', label='ERA5', rwidth=2)
plt.hist(np.mean(X0, axis=1), histtype='step', color='blue', label='CNRM-CM6-1', rwidth=2)
plt.hist(np.mean(Z0, axis=1), histtype='step', color='green', label='CNRM-CM6-1 bc', rwidth=2)
plt.title('Historical Train period (1980-1994) SBCK')
plt.legend()
plt.savefig(GRAPHS_DIR / 'test/exp3_sbck_train_hist_histo.png')

plt.figure(figsize=(6,6))
plt.hist(np.mean(Y1, axis=(1,2)), histtype='step', color='red', label='ERA5', rwidth=2)
plt.hist(np.mean(X1, axis=1), histtype='step', color='blue', label='CNRM-CM6-1', rwidth=2)
plt.hist(np.mean(Z1, axis=1), histtype='step', color='green', label='CNRM-CM6-1 bc', rwidth=2)
plt.title('Historical Test period (2000-2014) SBCK')
plt.legend()
plt.savefig(GRAPHS_DIR / 'test/exp3_sbck_test_hist_histo.png')

plt.figure(figsize=(6,6))
plt.hist(np.mean(X2, axis=1), histtype='step', color='blue', label='CNRM-CM6-1', rwidth=2)
plt.hist(np.mean(Z2, axis=1), histtype='step', color='green', label='CNRM-CM6-1 bc', rwidth=2)
plt.title('Future Test period (2015-2029) SBCK')
plt.legend()
plt.savefig(GRAPHS_DIR / 'test/exp3_sbck_test_future_histo.png')