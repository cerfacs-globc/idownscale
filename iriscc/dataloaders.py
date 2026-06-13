"""
Dataloader for the IRISCC dataset.

This module defines a custom PyTorch Dataset for loading and transforming the IRISCC dataset.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append(".")

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
import torch
from torch import Tensor

from iriscc.settings import get_train_split_dates
from iriscc.hparams import IRISCCHyperParameters
from iriscc.transforms import (MinMaxNormalisation,
                               LandSeaMask,
                               Pad,
                               FillMissingValue,
                               Log10Transform)

class IRISCC(Dataset):
    def __init__(self,
                 transform: v2.Compose | None,
                 hparams: IRISCCHyperParameters,
                 data_type: str = "train"):
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
        split_dates = get_train_split_dates(hparams.exp_name)

        list_data = np.sort([str(path) for path in self.sample_dir.glob("sample*")])
        train_start = np.where(list_data == str(self.sample_dir / f"sample_{split_dates[0]}.npz"))[0][0]
        val_start = np.where(list_data == str(self.sample_dir / f"sample_{split_dates[1]}.npz"))[0][0]
        test_start = np.where(list_data == str(self.sample_dir / f"sample_{split_dates[2]}.npz"))[0][0]

        if self.data_type == "train":
            self.samples = list_data[train_start:val_start-1]
        elif self.data_type == "val":
            self.samples = list_data[val_start:test_start-1]
        elif self.data_type == "test":
            self.samples = list_data[test_start:]

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

        Returns
        -------
            tuple[Tensor, Tensor]: Transformed input (x) and target (y) tensors.
        """
        data = dict(np.load(self.samples[idx], allow_pickle=True))
        x, y = data["x"], data["y"]
        if self.transform:
            x, y = self.transform((x, y))
        return x.float(), y.float()


def get_dataloaders(data_type: str, hparams: IRISCCHyperParameters | None = None) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader for the specified data type.

    Args:
        data_type (str): The type of data to load. Expected values are 'train' or other types
                            (e.g., 'validation', 'test'). Determines the shuffle behavior and batch size.

    Returns
    -------
        DataLoader: A PyTorch DataLoader object configured with the appropriate dataset,
                    transformations, batch size, and shuffle settings.
    """
    if hparams is None:
        hparams = IRISCCHyperParameters()
    transforms = v2.Compose([
                Log10Transform(hparams.channels),
                MinMaxNormalisation(hparams.statistics_dir, hparams.output_norm, hparams.output_range),
                LandSeaMask(hparams.mask, hparams.fill_value),
                FillMissingValue(hparams.fill_value),
                Pad(hparams.fill_value)
                ])
    training_data = IRISCC(transform=transforms,
                            hparams=hparams,
                            data_type=data_type)


    if data_type == "train":
        shuffle = True
    else:
        shuffle = False

    if data_type == "train":
        batch_size = hparams.batch_size
    else :
        batch_size = 1

    return DataLoader(training_data,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=1)

if __name__ == "__main__":
    train_dataloader = get_dataloaders("test")
    for batch in train_dataloader:
        x = batch[0][0, :, :, :]
        y = batch[1][0, :, :, :]
        y[y == -1.] = torch.nan

        print(x.shape, y.shape)
        print(y)

        break
