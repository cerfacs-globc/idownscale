import sys
sys.path.append('.')

import numpy as np
import glob
import json
import torch
import numpy as np
import xarray as xr
import torch.nn.functional as F

from iriscc.settings import (IMERG_MASK, CONFIG, DATASET_EXP4_30Y_DIR)
from iriscc.plotutils import plot_test

class StandardNormalisation():
    def __init__(self, sample_dir):
        statistics_file = sample_dir / 'statistics.json'
        with open(statistics_file) as f:
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
    def __init__(self, sample_dir, output_norm):

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
        self.output_norm = output_norm
    
    def __call__(self, sample):
        x, y = sample
        x = [(x[C,:,:] - self.min[C])/ (self.max[C] - self.min[C]) for C in range(len(x))]
        x = np.stack(x, axis=0)
        if self.output_norm is True:
            y[0,:,:] = (y[0,:,:] - self.min[-1])/ (self.max[-1] - self.min[-1])
        return torch.tensor(x), torch.tensor(y)

    
class DeMinMaxNormalisation():
    def __init__(self, sample_dir, output_norm):
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
        self.output_norm = output_norm
    
    def __call__(self, sample):
        x, y = sample
        if x is False:
            y[0,:,:] = y[0,:,:] * (self.max[-1] - self.min[-1]) + self.min[-1]
            return torch.tensor(y)  
        else:
            x = [x[C,:,:]*(self.max[C] - self.min[C]) + self.min[C] for C in range(len(x))]
            x = torch.stack(x, axis=0)
            if self.output_norm is True:
                y[0,:,:] = y[0,:,:] * (self.max[-1] - self.min[-1]) + self.min[-1]
            return torch.tensor(x), torch.tensor(y)
            
        

class LandSeaMask():
    def __init__(self, mask, fill_value):
        self.mask = mask
        if self.mask == 'continents':
            ds = xr.open_dataset(IMERG_MASK)
            ds['landseamask'].values = 100 - ds['landseamask'].values
            self.condition = ds['landseamask'].values < 25
        self.fill_value = fill_value

    def __call__(self, sample):
        x, y = sample
        if self.mask == 'none':
            pass

        else:
            if self.mask == 'target':
                self.condition = torch.isnan(y[0,:,:])
            for C in range(len(x)):
                x[C][self.condition] = self.fill_value

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
    
    
class DomainCrop():
    def __init__(self, sample_dir, domain_crop):
        self.domain = domain_crop
        self.sample_dir =sample_dir

    def __call__(self, sample):
        x, y = sample
        if self.domain is not None:
            coords_file = glob.glob(str(self.sample_dir/'coordinates.npz'))[0]
            coordinates = dict(np.load(coords_file, allow_pickle=True))
            self.lon = coordinates['lon']
            self.lat = coordinates['lat']
            lon_indices = np.where((self.lon >= self.domain[0]) & (self.lon <= self.domain[1]))[0]
            lat_indices = np.where((self.lat >= self.domain[2]) & (self.lat <= self.domain[3]))[0]
            
            lat_indices_torch = torch.tensor(lat_indices, dtype=torch.long)
            lon_indices_torch = torch.tensor(lon_indices, dtype=torch.long)
            x = x[:, lat_indices_torch][:, :, lon_indices_torch]
            y = y[:, lat_indices_torch][:, :, lon_indices_torch]
        return x, y

    
if __name__=='__main__':
    file = '/gpfs-calypso/scratch/globc/garcia/datasets/dataset_exp4_30y/sample_19850102.npz'
    coordinate = '/gpfs-calypso/scratch/globc/garcia/datasets/dataset_exp4_30y/coordinates.npz'
    data = dict(np.load(file, allow_pickle=True))
    x, y  = data['x'], data['y']
    domain = CONFIG['eobs']['domain']['france']
    crop = DomainCrop(sample_dir=DATASET_EXP4_30Y_DIR,
                      domain_crop=domain)
    x, y = crop((x,y))
    plot_test(y[0], 'y', '/gpfs-calypso/scratch/globc/garcia/graph/test.png')