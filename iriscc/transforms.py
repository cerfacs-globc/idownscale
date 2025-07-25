"""
Transformations class for the IRISCC dataset.
This module provides various transformations to preprocess the dataset for training and inference

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import numpy as np
import glob
import json
import torch
import numpy as np
import xarray as xr
import torch.nn.functional as F
from typing import Tuple, Union, List, Optional
from pathlib import Path

from iriscc.settings import IMERG_MASK, GRAPHS_DIR
from iriscc.plotutils import plot_test

class StandardNormalisation():
    """
    Applies Standard Normalisation applied to the data.
    """
    def __init__(self, sample_dir: Union[str, Path]) -> None:
        statistics_file = Path(sample_dir) / 'statistics.json'
        with open(statistics_file) as f:
            stats = json.load(f)
        mean = []
        std = []
        for channel in stats.keys():
            mean.append(stats[channel]['mean'])
            std.append(stats[channel]['std'])
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = sample
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        x = [(x[C, :, :] - self.mean[C]) / self.std[C] for C in range(len(x))]
        x = torch.stack(x, dim=0)
        return x, y
    

class MinMaxNormalisation():
    """
    Applies the Min-Max Normalisation applied to the data.
    """
    def __init__(self, sample_dir: Union[str, Path], output_norm: bool) -> None:
        statistics_file = Path(sample_dir) / 'statistics.json'
        with open(statistics_file) as f:
            stats = json.load(f)
        self.min = [stats[channel]['min'] for channel in stats.keys()]
        self.max = [stats[channel]['max'] for channel in stats.keys()]
        self.output_norm = output_norm
    
    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = sample
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        x = [(x[C, :, :] - self.min[C]) / (self.max[C] - self.min[C]) for C in range(len(x))]
        x = torch.stack(x, dim=0)
        if self.output_norm:
            y[0, :, :] = (y[0, :, :] - self.min[-1]) / (self.max[-1] - self.min[-1])
        return x, y

    
class DeMinMaxNormalisation:
    """
    Reverts the Min-Max Normalisation applied to the data.
    """
    def __init__(self, sample_dir: Union[str, Path], output_norm: bool) -> None:
        statistics_file = Path(sample_dir) / 'statistics.json'
        with open(statistics_file) as f:
            stats = json.load(f)
        self.min = [stats[channel]['min'] for channel in stats.keys()]
        self.max = [stats[channel]['max'] for channel in stats.keys()]
        self.output_norm = output_norm

    def __call__(self, sample: Tuple[Union[bool, torch.Tensor], torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, y = sample
        if x is False:
            y[0, :, :] = y[0, :, :] * (self.max[-1] - self.min[-1]) + self.min[-1]
            return torch.tensor(y)
        else:
            x = [x[C, :, :] * (self.max[C] - self.min[C]) + self.min[C] for C in range(len(x))]
            x = torch.stack(x, axis=0)
            if self.output_norm:
                y[0, :, :] = y[0, :, :] * (self.max[-1] - self.min[-1]) + self.min[-1]
            return torch.tensor(x), torch.tensor(y)
            
        
class LandSeaMask():
    """
    Applies a Nan mask to the data to match target land-sea mask.
    """
    def __init__(self, mask: str, fill_value: float) -> None:
        self.mask = mask
        if self.mask == 'continents':
            ds = xr.open_dataset(IMERG_MASK)
            ds['landseamask'].values = 100 - ds['landseamask'].values
            self.condition = ds['landseamask'].values < 25
        
        self.fill_value = fill_value

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = sample
        if self.mask == 'none':
            return x, y
        else:
            if self.mask == 'target':
                self.condition = torch.isnan(y[0, :, :])
            for C in range(len(x)):
                x[C][self.condition] = self.fill_value
        return x, y
    

class FillMissingValue():
    """
    Fills missing (NaN) values in the data with a specified fill value.
    """
    def __init__(self, fill_value: float) -> None:
        self.fill_value = fill_value

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = sample
        x = torch.where(torch.isnan(x), torch.tensor(self.fill_value, dtype=x.dtype, device=x.device), x)
        y = torch.where(torch.isnan(y), torch.tensor(self.fill_value, dtype=y.dtype, device=y.device), y)
        return x, y


class Pad:
    """
    A class to apply padding to 2D arrays to make their dimensions divisible by a specified divisor.

    Attributes:
        fill_value (float): The value to use for padding.
        divisor (int): The divisor to which the dimensions of the array will be aligned. Defaults to 32.
    """
    def __init__(self, fill_value: float) -> None:
        self.divisor: int = 32
        self.fill_value: float = fill_value

    def pad_func(self, array: torch.Tensor) -> torch.Tensor:
        H, W = array.shape
        new_H = ((H + self.divisor - 1) // self.divisor) * self.divisor
        new_W = ((W + self.divisor - 1) // self.divisor) * self.divisor

        padding_H = new_H - H
        padding_W = new_W - W

        padding = (padding_W // 2, padding_W - padding_W // 2,
                   padding_H // 2, padding_H - padding_H // 2)

        padded_array = F.pad(array, padding, mode='constant', value=self.fill_value)
        return padded_array

    def __call__(self, sample: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = sample
        x = [self.pad_func(x[C]) for C in range(len(x))]
        x = torch.stack(x)
        y = [self.pad_func(y[C]) for C in range(len(y))]
        y = torch.stack(y)
        return x, y

        
class UnPad():
    class UnPad:
        """
        A class to remove padding from arrays or tensors based on the initial size.

        Attributes:
            initial_size (Tuple[int, int]): The target height and width of the unpadded array.
            divisor (int): A constant divisor used for padding calculations (default is 32).
        """
    def __init__(self, initial_size: Tuple[int, int]) -> None:
        self.divisor: int = 32
        self.initial_size: Tuple[int, int] = initial_size

    def unpad_func(self, array: torch.Tensor) -> torch.Tensor:
        H, W = self.initial_size[0], self.initial_size[1]

        new_H, new_W = array.shape

        padding_H = new_H - H
        padding_W = new_W - W

        pad_top = padding_H // 2
        pad_left = padding_W // 2
        unpadded_array = array[pad_top:pad_top + H, pad_left:pad_left + W]
        return unpadded_array

    def __call__(self, sample: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        y = sample
        y = [self.unpad_func(y[C]) for C in range(len(y))]
        y = torch.stack(y)
        return y
    
    
class DomainCrop:
    """
    Crops the input data to a specified domain based on latitude and longitude.

    Attributes:
        sample_dir (Union[str, Path]): Directory containing the coordinates file.
        domain_crop (Optional[Tuple[float, float, float, float]]): The cropping domain specified as (lon_min, lon_max, lat_min, lat_max).
    """
    def __init__(self, sample_dir: Union[str, Path], domain_crop: Optional[Tuple[float, float, float, float]]) -> None:
        self.sample_dir = Path(sample_dir)
        self.domain = domain_crop

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = sample
        if self.domain is not None:
            coords_file = glob.glob(str(self.sample_dir / 'coordinates.npz'))[0]
            coordinates = dict(np.load(coords_file, allow_pickle=True))
            self.lon = coordinates['lon']
            self.lat = coordinates['lat']
            lon_indices = np.where((self.lon >= self.domain[0]) & (self.lon <= self.domain[1]))[0]
            lat_indices = np.where((self.lat >= self.domain[2]) & (self.lat <= self.domain[3]))[0]
            
            lat_indices_torch = torch.tensor(lat_indices, dtype=torch.long)
            lon_indices_torch = torch.tensor(lon_indices, dtype=torch.long)
            x = x[:, lat_indices_torch][:, :, lon_indices_torch]
            y = y[:, lat_indices_torch][:, :, lon_indices_torch]
        return torch.tensor(x), torch.tensor(y)

class Log10Transform:
    """
    Applies a logarithmic transformation to the input data, specifically for the precipitations channel.
    """
    def __init__(self, channels:List[str]):
        self.channels = channels

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = sample
        x, y = torch.tensor(x), torch.tensor(y)
        if 'pr input' in self.channels:
            pr_i = self.channels.index('pr input')
            x[pr_i, :, :] = torch.log10(1 + x[pr_i, :, :])
        return x, y

