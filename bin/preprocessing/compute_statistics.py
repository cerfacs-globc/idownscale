import sys
sys.path.append('.')

import numpy as np
import glob
import json

from iriscc.settings import DATASET_EXP1_DIR, CHANELS

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


if __name__=='__main__':
    dataset = glob.glob(str(DATASET_EXP1_DIR/'sample*'))
    ch = len(CHANELS)
    sum = np.zeros(ch)
    square_sum = np.zeros(ch)
    n_total = np.zeros(ch)

    for nb, sample in enumerate(dataset):
        data = dict(np.load(sample, allow_pickle=True))
        x, y = data['x'], data['y']
        condition = np.isnan(y[0])
        x[condition] = np.nan
      

        if nb == 0:
            min, max = np.nanmin(x, axis=(1, 2)), np.nanmax(x, axis=(1, 2))
        for i in range(ch):
            sum[i], square_sum[i], n_total[i], min[i], max[i] = update_statistics(sum[i], 
                                                                    square_sum[i], 
                                                                    n_total[i],
                                                                    min[i],
                                                                    max[i],
                                                                    x[i])
    mean = sum / n_total
    std = np.sqrt((square_sum / n_total) - (mean**2))
    print(mean, std, min, max)

    stats = {}
    for i, chanel in enumerate(CHANELS):
        stats[chanel] = {'mean': mean[i],
                         'std': std[i],
                         'min': min[i],
                         'max': max[i]}

    with open(DATASET_EXP1_DIR/'statistics.json', "w") as f: 
	    json.dump(stats, f)