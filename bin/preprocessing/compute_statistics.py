"""
Compute statistics for a dataset and plot histograms for the inpus and target channels.
Save the statistics in a JSON file and the histograms as PNG files.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import numpy as np
import glob
import json
import argparse
import matplotlib.pyplot as plt

from iriscc.settings import CONFIG, DATES_TRAIN
from typing import Tuple

def update_statistics(sum: float, 
                      square_sum: float, 
                      n_total: int, 
                      min: float, 
                      max: float, 
                      x: np.ndarray) -> Tuple[float, float, int, float, float]:
    ''' 
    Compute and update sample statistics including sum, squared sum, total count, 
    minimum, and maximum values for a given array, ignoring NaN values.
    '''
    x = x[~np.isnan(x)]  # Remove NaN values
    sum += np.sum(x)  # Update sum
    square_sum += np.sum(x**2)  # Update squared sum
    n_total += x.size  # Update total count
    if np.min(x) < min:  # Update minimum value
         min = np.min(x)
    if np.max(x) > max:  # Update maximum value
         max = np.max(x)
    return sum, square_sum, n_total, min, max


def plot_histogram(data, 
                   min:float, 
                   max:float, 
                   mean:float, 
                   std:float, 
                   var:str, 
                   title:str, 
                   save_dir:str):
    
    hist, edges = np.histogram(data, bins=50, range=(min, max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(10, 6))
    plt.bar(centers, hist, align='center', width=np.diff(edges), alpha=0.5, color='blue', label='Density')
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'$\mu$ = {mean:.2f}')
    plt.axvline(mean - std, color='green', linestyle='--', linewidth=2, label=f' $\sigma$ = {std:.2f}')
    plt.axvline(mean + std, color='green', linestyle='--', linewidth=2)

    plt.xlabel(var, fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=14)
    plt.savefig(save_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Compute statistics for a given dataset path")
    parser.add_argument('--exp', type=str, help='Experiment name', default='exp6')  
    args = parser.parse_args()

    dataset_dir = CONFIG[args.exp]['dataset']
    dataset = np.sort(glob.glob(str(dataset_dir/'sample*')))
    channels = CONFIG[args.exp]['channels']
    ch = len(channels)
    sum = np.zeros(ch)
    square_sum = np.zeros(ch)
    n_total = np.zeros(ch)

    # Other way to split the dataset
    #nb = len(dataset)
    #train_end = int(0.6 * nb) 
    #val_end = train_end + int(0.2 * nb)
    
    train_start = np.where(dataset == str(dataset_dir/f'sample_{DATES_TRAIN[0]}0101.npz'))[0][0]
    val_start = np.where(dataset == str(dataset_dir/f'sample_{DATES_TRAIN[1]}0101.npz'))[0][0]
    test_start = np.where(dataset == str(dataset_dir/f'sample_{DATES_TRAIN[2]}0101.npz'))[0][0]
   
    y_data = {'train' : [],
                 'val' : [],
                 'test' : []}

    for nb, sample in enumerate(dataset):
        print(sample)

        data = dict(np.load(sample, allow_pickle=True))
        x, y = data['x'], data['y']
        condition = np.isnan(y[0])
        for c, channel in enumerate(channels[:-1]):  # Exclude the last channel (target)
            x[c][condition] = np.nan
            if channel == 'pr input':
                x[c][np.isnan(x[c])] = 0.
                x[c] = np.log10(1 + x[c])  # Apply log transformation to precipitation input
                x[c][condition] = np.nan
                print(np.nanmax(x[c]))

        # Only training statistics are used for normalization
        if nb in range(train_start, val_start-1):
            y_data['train'].append(y.flatten())
            if nb == train_start:
                min, max = np.nanmin(x, axis=(1, 2)), np.nanmax(x, axis=(1, 2))
                min = np.concatenate((min, np.nanmin(y, axis=(1, 2))))
                max = np.concatenate((max, np.nanmax(y, axis=(1, 2))))
            
            for i in range(ch):
                if i == ch-1:
                    sum[i], square_sum[i], n_total[i], min[i], max[i] = update_statistics(sum[i], 
                                                                        square_sum[i], 
                                                                        n_total[i],
                                                                        min[i],
                                                                        max[i],
                                                                        y[0])
                else:
                    sum[i], square_sum[i], n_total[i], min[i], max[i] = update_statistics(sum[i], 
                                                                        square_sum[i], 
                                                                        n_total[i],
                                                                        min[i],
                                                                        max[i],
                                                                        x[i])
                


        # Validation and test data histograms 
        elif nb in range(val_start, test_start-1):
            y_data['val'].append(y.flatten())
        elif nb >= test_start :
            y_data['test'].append(y.flatten())
        
    mean = sum / n_total
    std = np.sqrt((square_sum / n_total) - (mean**2))

    stats = {}
    for i, chanel in enumerate(channels):
        stats[chanel] = {'mean': mean[i],
                         'std': std[i],
                         'min': min[i].astype(np.float64),
                         'max': max[i].astype(np.float64)}
    
    with open(dataset_dir/'statistics.json', "w") as f: 
        json.dump(stats, f)

    for type, data in y_data.items():
        data = np.concatenate(data)
        plot_histogram(data, 
                    np.nanmin(data), 
                    np.nanmax(data), 
                    np.nanmean(data), 
                    np.nanstd(data), 
                    CONFIG[args.exp]['target_vars'][0], 
                    f'y {type} dataset histogram', 
                    dataset_dir/f'hist_y_{type}.png')
 
    

