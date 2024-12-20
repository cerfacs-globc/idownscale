import sys
sys.path.append('.')

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
import glob

from iriscc.hparams import IRISCCHyperParameters
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue

class IRISCC(Dataset):
    def __init__(self,
                 transform,
                 hparams,
                 data_type='train'):
        
        self.sample_dir = hparams.sample_dir
        self.transform = transform
        self.data_type = data_type

        list_data = np.sort(glob.glob(str(self.sample_dir/'sample*')))
        nb = len(list_data)
        train_end = int(0.6 * nb) 
        val_end = train_end + int(0.2 * nb)

        if self.data_type == 'train':
            self.samples = list_data[:train_end]
        elif self.data_type =='val':
            self.samples = list_data[train_end:val_end]
        elif self.data_type == 'test':
            self.samples = list_data[val_end:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = dict(np.load(self.samples[idx]), allow_pickle=True)
        x, y = data['x'], data['y']
        if self.transform:
                x, y = self.transform((x, y))
        return x.float(), y.float()


def get_dataloaders(data_type):

    hparams = IRISCCHyperParameters()
    transforms = v2.Compose([
            MinMaxNormalisation(), 
            LandSeaMask(hparams.mask, hparams.fill_value, hparams.landseamask),
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
                            num_workers=4)
    return dataloader   

if __name__=='__main__':
    train_dataloader = get_dataloaders('train')
    for batch in train_dataloader:
        print(batch[0].shape)