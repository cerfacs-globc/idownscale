"""
Dataloader for the IRISCC dataset.
This module defines a custom PyTorch Dataset for loading and transforming the IRISCC dataset.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
import torch
import glob
from typing import Optional
from torch import Tensor

from iriscc.settings import GRAPHS_DIR
from iriscc.plotutils import plot_test
from iriscc.hparams import IRISCCHyperParameters
from iriscc.transforms import (MinMaxNormalisation, 
                               LandSeaMask, 
                               Pad, 
                               FillMissingValue,
                               Log10Transform)

class IRISCC(Dataset):
    def __init__(self,
                 transform: Optional[v2.Compose],
                 hparams: IRISCCHyperParameters,
                 data_type: str = 'train'):
        """
        A custom PyTorch Dataset for loading and transforming IRISCC data.

        Args:
            transform (Optional[v2.Compose]): Transformations to apply to the data.
            hparams (IRISCCHyperParameters): Hyperparameters for the dataset.
            data_type (str): Type of data to load ('train', 'val', or 'test').
        """
        self.sample_dir = hparams.sample_dir
        self.transform = transform
        self.data_type = data_type

        list_data = np.sort(glob.glob(str(self.sample_dir / 'sample*')))
        #train_end = np.where(list_data == str(self.sample_dir / 'sample_20091231.npz'))[0][0]
        #val_end = np.where(list_data == str(self.sample_dir / 'sample_20131231.npz'))[0][0]
        train_start = np.where(list_data == str(self.sample_dir / 'sample_19850101.npz'))[0][0]
        train_end = np.where(list_data == str(self.sample_dir / 'sample_20041231.npz'))[0][0]
        val_end = np.where(list_data == str(self.sample_dir / 'sample_20091231.npz'))[0][0]

        if self.data_type == 'train':
            self.samples = list_data[train_start:train_end]
        elif self.data_type == 'val':
            self.samples = list_data[train_end:val_end]
        elif self.data_type == 'test':
            self.samples = list_data[val_end:]

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, Tensor]: Transformed input (x) and target (y) tensors.
        """
        data = dict(np.load(self.samples[idx], allow_pickle=True))
        x, y = data['x'], data['y']
        if self.transform:
            x, y = self.transform((x, y))
        return x.float(), y.float()


def get_dataloaders(data_type: str) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader for the specified data type.
    Args:
        data_type (str): The type of data to load. Expected values are 'train' or other types 
                            (e.g., 'validation', 'test'). Determines the shuffle behavior and batch size.
    Returns:
        DataLoader: A PyTorch DataLoader object configured with the appropriate dataset, 
                    transformations, batch size, and shuffle settings.
    """

    hparams = IRISCCHyperParameters()
    transforms = v2.Compose([
                Log10Transform(hparams.channels),
                MinMaxNormalisation(hparams.sample_dir, hparams.output_norm), 
                LandSeaMask(hparams.mask, hparams.fill_value),
                FillMissingValue(hparams.fill_value),
                Pad(hparams.fill_value)
                ])
    training_data = IRISCC(transform=transforms,
                        hparams=hparams,
                        data_type=data_type)
    
    
    if data_type == 'train':
        shuffle=True
    else:
        shuffle=False

    if data_type == 'train':
        batch_size = hparams.batch_size
    else : 
        batch_size = 1

    dataloader = DataLoader(training_data, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=1)
    return dataloader   

if __name__=='__main__':
    train_dataloader = get_dataloaders('train')
    for batch in train_dataloader:
        x = batch[0][0,:,:,:]
        y = batch[1][0,:,:,:]
        
        print(x.shape, y.shape)
        print(y)
        plot_test(x[1].numpy(), GRAPHS_DIR/'test.png', title='huss')
        plot_test(x[2].numpy(), GRAPHS_DIR/'test1.png', title='psl')
        plot_test(x[3].numpy(), GRAPHS_DIR/'test2.png', title='tas')
        plot_test(y[0].numpy(), GRAPHS_DIR/'test3.png', title='pr y')
        break