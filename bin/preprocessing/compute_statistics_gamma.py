""" 
Compute alpha and beta statistics (Gamma distribution) on training historical period 

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import numpy as np
import glob
import argparse
from scipy.stats import gamma

from iriscc.settings import CONFIG

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Compute statistics for a given dataset path")
    parser.add_argument('--exp', type=str, help='Experiment name', default='exp6')  
    args = parser.parse_args()

    dataset_dir = CONFIG[args.exp]['dataset']
    input_vars = CONFIG[args.exp]['input_vars']
    pr_i = np.where(np.array(input_vars) == 'pr')[0][0] if 'pr' in input_vars else None

    h, w = CONFIG[args.exp]['shape']

    alpha, beta = [], []
    for nb, year in enumerate(np.arange(1980, 2009)):

        files = np.sort(glob.glob(str(dataset_dir/f'sample_{year}*')))
        x_year = []
        for file in files:
            data = dict(np.load(file, allow_pickle=True))
            x = data['x'][pr_i]  # Select the precipitation input variable
            x_year.append(x)
        x_year = np.stack(x_year, axis=0)
        x_year = x_year.reshape(-1, h * w)

        alpha_year, beta_year = [], []
        for cell in range(h * w):
            x_year_cell = x_year[:, cell] # precip time series for a single cell
            x_year_cell = x_year_cell[x_year_cell > 1] # Only rainy days
            if len(x_year_cell) == 0:  # For sea cells or cells with no precipitation
                alpha_cell, beta_cell = 0, 0
            else:
                alpha_cell, beta_cell, _ = gamma.fit(x_year_cell.flatten())
            alpha_year.append(alpha_cell)
            beta_year.append(beta_cell)
        alpha.append(np.array(alpha_year).reshape(h, w))
        beta.append(np.array(beta_year).reshape(h, w))
    alpha = np.mean(np.array(alpha), axis=0)
    beta = np.mean(np.array(beta), axis=0)
 
    params = {'alpha': alpha, 'beta': beta}
    np.savez(str(dataset_dir/'gamma_params.npz'), **params)
            

