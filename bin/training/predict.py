import sys
sys.path.append('.')

import glob
import torch
import xarray as xr
import numpy as np
from torchvision.transforms import v2

from iriscc.hparams import IRISCCHyperParameters
from iriscc.lightning_module import IRISCCLightningModule
from iriscc.plotutils import plot_test
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import DATASET_EXP1_DIR, GRAPHS_DIR, TARGET_SIZE, TARGET_GRID_FILE

def postprocess_pred(y):
    # unpad
    unpad_func = UnPad(TARGET_SIZE)
    y = unpad_func(y)
    
    # to numpy
    y = y[0,:,:].numpy()

    # france mask
    ds = xr.open_dataset(TARGET_GRID_FILE)
    ds = ds.isel(time=0)
    condition = np.isnan(ds['tas'].values)
    y[condition] = np.nan
    return y



if __name__=='__main__':
    date = str(sys.argv[1])
    checkpoint_dir = str(sys.argv[2])
    hparams = IRISCCHyperParameters()
    model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir)
    model.eval()

    transforms = v2.Compose([
                MinMaxNormalisation(), 
                LandSeaMask(hparams.mask, hparams.fill_value, hparams.landseamask),
                FillMissingValue(hparams.fill_value),
                Pad(hparams.fill_value)
                ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sample = glob.glob(str(DATASET_EXP1_DIR/f'sample_{date}.npz'))[0]
    data = dict(np.load(sample), allow_pickle=True)
    x, y = data['x'], data['y']
    x, _ = transforms((x, y))

    x = torch.unsqueeze(x, dim=0).float()
    y_hat = model(x.to(device)).to(device)
    y_hat = y_hat.detach().cpu()
    y_hat_raw = y_hat[0,0,:,:].numpy()
    #y_hat_raw[x[0,0,:,:] ==  hparams.fill_value] = np.nan
    plot_test(y_hat_raw, 'y_hat_raw', GRAPHS_DIR/f'pred/{date}_yhatraw.png')
    
    y_hat = postprocess_pred(y_hat.detach().cpu()[0])
    
    plot_test(y_hat, 'y_hat', GRAPHS_DIR/f'pred/{date}_yhat.png')
    plot_test(y[0], 'y', GRAPHS_DIR/f'pred/{date}_y.png')
    plot_test(y[0]-y_hat, 'y-y_hat', GRAPHS_DIR/f'pred/{date}_diff.png')

