import sys
sys.path.append('.')

import numpy as np
import json
import torch
import xarray as xr
import torch.nn.functional as F

from iriscc.plotutils import plot_test
from iriscc.settings import (TARGET_GRID_FILE, IMERG_MASK)

class StandardNormalisation():
    def __init__(self):
        with open(STATISTICS_FILE) as f:
            stats = json.load(f)
        mean = []
        std = []
        for chanel in stats.keys():
            mean.append(stats[chanel]['mean'])
            std.append(stats[chanel]['std'])
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        x, y = sample
        x = [(x[C,:,:] - self.mean[C])/ self.std[C] for C in range(len(x))]
        x = np.stack(x, axis=0)
        return torch.tensor(x), torch.tensor(y)
        

class MinMaxNormalisation():
    def __init__(self, sample_dir):
        statistics_file = sample_dir / 'statistics.json'
        with open(statistics_file) as f:
            stats = json.load(f)
        min = []
        max = []
        for chanel in stats.keys():
            min.append(stats[chanel]['min'])
            max.append(stats[chanel]['max'])
        self.min = min
        self.max = max
    
    def __call__(self, sample):
        x, y = sample
        x = [(x[C,:,:] - self.min[C])/ (self.max[C] - self.min[C]) for C in range(len(x))]
        x = np.stack(x, axis=0)
        return torch.tensor(x), torch.tensor(y)

    
class LandSeaMask():
    def __init__(self, mask, fill_value):
        self.mask = mask
        if self.mask == 'france':
            ds = xr.open_dataset(TARGET_GRID_FILE)
            ds = ds.isel(time=0)
            self.condition = np.isnan(ds['tas'].values)
        elif self.mask == 'continents':
            ds = xr.open_dataset(IMERG_MASK)
            ds['landseamask'].values = 100 - ds['landseamask'].values
            self.condition = ds['landseamask'].values < 25
        self.fill_value = fill_value

    def __call__(self, sample):
        x, y = sample
        if self.mask == 'none':
            pass
        else:
            for C in range(len(x)):
                x[C][self.condition] = self.fill_value

            mask = (~self.condition).astype(int)
            mask = np.expand_dims(mask, axis=0)
            x = torch.concatenate([x, torch.tensor(mask)], axis=0)
        return x, y
    
class FillMissingValue():
    def __init__(self, fill_value):
        self.fill_value = fill_value
    def __call__(self, sample):
        x, y = sample
        x[torch.isnan(x)] = self.fill_value
        y[torch.isnan(y)] = self.fill_value
        return x, y


class Pad():
    def __init__(self, fill_value):
        self.divisor = 32
        self.fill_value = fill_value

    def pad_func(self, array):
        H, W = array.shape
        new_H = ((H + self.divisor - 1) // self.divisor) * self.divisor
        new_W = ((W + self.divisor - 1) // self.divisor) * self.divisor

        padding_H = new_H - H
        padding_W = new_W - W

        padding = (padding_W // 2, padding_W - padding_W // 2,
        padding_H // 2, padding_H - padding_H // 2)

        padded_array = F.pad(array, padding, mode='constant', value=self.fill_value)
        return padded_array

    def __call__(self, sample):
        x, y = sample
        x = [self.pad_func(x[C]) for C in range(len(x))]
        x = torch.stack(x)
        y = [self.pad_func(y[C]) for C in range(len(y))]
        y = torch.stack(y)
        return x, y

        
class UnPad():
    def __init__(self, initial_size):
        self.divisor = 32
        self.initial_size = initial_size

    def unpad_func(self, array):
        H, W = self.initial_size[0], self.initial_size[1]

        new_H, new_W = array.shape 

        padding_H = new_H - H
        padding_W = new_W - W

        pad_top = padding_H // 2
        pad_left = padding_W // 2
        unpadded_array = array[pad_top:pad_top + H, pad_left:pad_left + W]
        return unpadded_array

    def __call__(self, sample):
        y = sample
        y = [self.unpad_func(y[C]) for C in range(len(y))]
        y = torch.stack(y)
        return y

    
if __name__=='__main__':
    
    data = dict(np.load('/gpfs-calypso/scratch/globc/garcia/datasets/dataset_exp1/sample_20141229.npz', allow_pickle=True))
    sample = (data['x'], data['y'])
    x, y = sample
    
    
    # test norm
    norm = MinMaxNormalisation()
    x, y = norm(sample)

    
    # test nan
    nan = LandSeaMask('france', fill_value=-1.)
    x, y = nan((x, y))
    print(x.shape)

    # test resize 
    #resize = Pad()
    #x, y = resize((torch.tensor(x), torch.tensor(y)))

    #unpad = UnPad(initial_size=[134,143])
    #y = unpad(y)

    #x[x == -9999] = np.nan
    #y[y == -9999] = np.nan
    #x[np.isnan(x) == 1000000]
    
    #print('y',np.argwhere(np.isnan(y)))

    #plot_test(np.flip(y[0].numpy(), axis=0), 'title', '/gpfs-calypso/scratch/globc/garcia/graph/datasets/testnan.png')

