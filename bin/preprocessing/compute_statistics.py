import sys
sys.path.append('.')

import numpy as np
import glob
import json
import matplotlib.pyplot as plt

from iriscc.settings import DATASET_EXP1_DIR, CHANELS, DATASET_EXP1_30Y_DIR

def update_statistics(sum, square_sum, n_total, min, max, x):
    ''' Compute and update samples statistics '''
    x = x[~np.isnan(x)]
    sum += np.sum(x)
    square_sum += np.sum(x**2)
    n_total += x.size
    if np.min(x) < min:
         min = np.min(x)
    if np.max(x) > max:
         max = np.max(x)
    return sum, square_sum, n_total, min, max


def plot_histogram(data, min, max, mean, std, variable:str, title:str, save_dir:str):
    hist, edges = np.histogram(data, bins=50, range=(min, max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(10, 6))
    plt.bar(centers, hist, align='center', width=np.diff(edges), alpha=0.5, color='blue', label='Density')
    print(mean, std)
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'$\mu$ = {mean:.2f}')
    plt.axvline(mean - std, color='green', linestyle='--', linewidth=2, label=f' $\sigma$ = {std:.2f}')
    plt.axvline(mean + std, color='green', linestyle='--', linewidth=2)

    plt.xlabel(variable, fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=14)
    plt.savefig(save_dir)


if __name__=='__main__':
    dataset = np.sort(glob.glob(str(DATASET_EXP1_30Y_DIR/'sample*')))
    ch = len(CHANELS)
    sum = np.zeros(ch)
    square_sum = np.zeros(ch)
    n_total = np.zeros(ch)

    nb = len(dataset)
    train_end = int(0.6 * nb) 
    val_end = train_end + int(0.2 * nb)
    print(dataset[val_end])
    
    x_data = {'train' : [],
                 'val' : [],
                 'test' : []}
    y_data = {'train' : [],
                 'val' : [],
                 'test' : []}

    for nb, sample in enumerate(dataset):
        print(nb)
        data = dict(np.load(sample, allow_pickle=True))
        x, y = data['x'], data['y']
        condition = np.isnan(y[0])
        for c in range(len(x)):
            x[c][condition] = np.nan

        if nb == 0:
            min, max = np.nanmin(x, axis=(1, 2)), np.nanmax(x, axis=(1, 2))
        for i in range(ch):
            sum[i], square_sum[i], n_total[i], min[i], max[i] = update_statistics(sum[i], 
                                                                    square_sum[i], 
                                                                    n_total[i],
                                                                    min[i],
                                                                    max[i],
                                                                    x[i])
        if nb < train_end :
            x_data['train'].append(x[1:,:,:].flatten())
            y_data['train'].append(y.flatten())
        elif nb in range(train_end, val_end):
            x_data['val'].append(x[1:,:,:].flatten())
            y_data['val'].append(y.flatten())
        elif nb >= val_end :
            x_data['test'].append(x[1:,:,:].flatten())
            y_data['test'].append(y.flatten())
        
    mean = sum / n_total
    std = np.sqrt((square_sum / n_total) - (mean**2))
    print(mean, std, min, max)

    stats = {}
    for i, chanel in enumerate(CHANELS):
        stats[chanel] = {'mean': mean[i],
                         'std': std[i],
                         'min': min[i],
                         'max': max[i]}

    for name, dict in {'x':x_data, 'y':y_data}.items():
        for type, data in dict.items():
            ''' Only Temperature values are kept '''
            data = np.concatenate(data)
            plot_histogram(data, 
                        min[1:].min(), 
                        max[1:].max(), 
                        np.nanmean(data), 
                        np.nanstd(data), 
                        'tas (K)', 
                        f'{name} {type} dataset histogram', 
                        DATASET_EXP1_30Y_DIR/f'hist_{name}_{type}.png')
         


    with open(DATASET_EXP1_30Y_DIR/'statistics.json', "w") as f: 
	    json.dump(stats, f)
