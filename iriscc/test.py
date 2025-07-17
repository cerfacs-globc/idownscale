import sys

import scipy.stats
sys.path.append('.')

import os
import glob
import torch
import pyproj
import xarray as xr
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch.distributions.gamma import Gamma
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy
from torchmetrics import MeanSquaredError, PearsonCorrCoef

from iriscc.settings import (GRAPHS_DIR, 
                             TARGET_EOBS_FRANCE_FILE, 
                             CONFIG, 
                             TARGET_GCM_FILE, 
                             RCM_RAW_DIR, 
                             ALADIN_PROJ_PYPROJ,
                             DATES_BC_TEST_HIST,
                             DATES_BC_TEST_FUTURE,
                             DATASET_BC_DIR,
                             PREDICTION_DIR,
                             GCM_RAW_DIR,
                             DATASET_EXP6_30Y_DIR
                             )
from iriscc.transforms import DomainCrop
from iriscc.plotutils import plot_map_contour, plot_test
from iriscc.datautils import reformat_as_target, standardize_longitudes, Data, interpolation_target_grid, crop_domain_from_ds
from bin.training.predict_loop import get_target_format

# Generate 100 curves transitioning from pdf1 to pdf2
'''
file = '/gpfs-calypso/scratch/globc/garcia/datasets/dataset_bc/bc_test_future_rcm.npz'
data = dict(np.load(file, allow_pickle=True))
dates = data['dates']
rcm = data['rcm']
rcm = rcm[:, 4:11, 4:11]  # Crop to the region of interest
index = np.where(dates == pd.Timestamp('2070-01-01'))[0]
plot_test(rcm[index[0],:,:], GRAPHS_DIR/'test.png')

index = np.where(dates == pd.Timestamp('2030-01-01'))[0]
plot_test(rcm[index[0],:,:], GRAPHS_DIR/'test1.png')
'''
file = '/gpfs-calypso/scratch/globc/garcia/datasets/dataset_bc/dataset_exp5_test_rcm_bc/sample_20150101.npz'
data = dict(np.load(file, allow_pickle=True))
x = data['x']
plot_test(x[1], GRAPHS_DIR/'test.png')

file = '/gpfs-calypso/scratch/globc/garcia/datasets/dataset_bc/dataset_exp5_test_rcm_bc/sample_20350101.npz'
data = dict(np.load(file, allow_pickle=True))
x = data['x']
plot_test(x[1], GRAPHS_DIR/'test1.png')

file = '/gpfs-calypso/scratch/globc/garcia/datasets/dataset_bc/dataset_exp5_test_rcm_bc/sample_20800101.npz'
data = dict(np.load(file, allow_pickle=True))
x = data['x']
plot_test(x[1], GRAPHS_DIR/'test2.png')

file = '/gpfs-calypso/scratch/globc/garcia/datasets/dataset_bc/dataset_exp5_test_gcm_bc/sample_19800107.npz'
data = dict(np.load(file, allow_pickle=True))
x = data['x']
plot_test(x[1], GRAPHS_DIR/'test3.png')
