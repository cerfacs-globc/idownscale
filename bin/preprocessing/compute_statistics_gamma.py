"""
Compute alpha and beta statistics (Gamma distribution) on training historical period
If target variable is precipitation

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import numpy as np
import glob
import argparse
from scipy.stats import gamma

from iriscc.settings import CONFIG, DATES_TRAIN

def filter_aberrant_values(param: np.ndarray, threshold: float) -> np.ndarray:
    """
    Filter aberrant values in the param 3D array based on a threshold.
    Replace aberrant values with the mean of neighboring cells.
    """
    indices = np.where(param > threshold)
    
    for t, i, j in zip(*indices):
        param_cell = np.mean(param[t,max(0, i-1):min(param.shape[1], i+2), 
                                   max(0, j-1):min(param.shape[2], j+2)])
        param[t, i, j] = param_cell
    
    return param

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Compute statistics for a given dataset path")
    parser.add_argument('--exp', type=str, help='Experiment name', default='exp6')  
    args = parser.parse_args()

    dataset_dir = CONFIG[args.exp]['dataset']
    input_vars = CONFIG[args.exp]['input_vars']

    h, w = CONFIG[args.exp]['shape']

    alpha, beta = [], []
    calibration_y = 1
    for nb in range(int(DATES_TRAIN[0]), int(DATES_TRAIN[1])-1, calibration_y):  # Increment by 5 years
        y_year = []
        for year in range(nb, nb + calibration_y):  # Collect data for 5 years
            files = np.sort(glob.glob(str(dataset_dir/f'sample_{year}*')))
            for file in files:
                data = dict(np.load(file, allow_pickle=True))
                y = data['y'][0]  # Select the precipitation input variable
                y_year.append(y)
        y_year = np.stack(y_year, axis=0)
        y_year = y_year.reshape(-1, h * w)  # Reshape to (365*5, h*w)

        alpha_year, beta_year = [], []
        for cell in range(h * w):
            y_year_cell = y_year[:, cell] # precip time series for a single cell
            y_year_cell = y_year_cell[y_year_cell > 1] # Only rainy days
            if len(y_year_cell) == 0:  # For sea cells or cells with no precipitation
                alpha_cell, beta_cell = np.nan, np.nan
            else:
                shape_cell, _, scale_cell = gamma.fit(y_year_cell.flatten())
                alpha_cell, beta_cell = shape_cell, 1/scale_cell

            alpha_year.append(alpha_cell)
            beta_year.append(beta_cell)
        alpha.append(np.array(alpha_year).reshape(h, w))
        beta.append(np.array(beta_year).reshape(h, w))

    # Filter aberrant values
    alpha = np.array(alpha)
    beta = np.array(beta)
    alpha = filter_aberrant_values(alpha, 
                                   8*np.nanstd(alpha, where=alpha>0)) 
    beta = filter_aberrant_values(beta,
                                      8*np.nanstd(beta, where=beta>0))

    alpha = np.nanmean(alpha, axis=0)
    beta = np.nanmean(beta, axis=0)
    print(beta,alpha)
    
    params = {'alpha': alpha, 'beta': beta}
    np.savez(str(dataset_dir/'gamma_params.npz'), **params)
            

