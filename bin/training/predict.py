import sys
sys.path.append('.')

import glob
import torch
import xarray as xr
import numpy as np
from torchvision.transforms import v2

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.plotutils import plot_test, plot_contour
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import GRAPHS_DIR, TARGET_SIZE, RUNS_DIR


if __name__=='__main__':
    date = str(sys.argv[1])
    exp = str(sys.argv[2]) # ex : exp 1
    test_name = str(sys.argv[3]) # ex : mask_continents
    version = str(sys.argv[4])
    run_dir = RUNS_DIR/f'{exp}/{test_name}/lightning_logs/version_{version}'
    checkpoint_dir = run_dir/'checkpoints/best-checkpoint.ckpt'

    model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir)
    model.eval()
    hparams = model.hparams['hparams']
    arch = hparams['model']
    transforms = v2.Compose([
                MinMaxNormalisation(), 
                LandSeaMask(hparams['mask'], hparams['fill_value']),
                FillMissingValue(hparams['fill_value']),
                Pad(hparams['fill_value'])
                ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sample = glob.glob(str(hparams['sample_dir']/f'sample_{date}.npz'))[0]
    data = dict(np.load(sample), allow_pickle=True)
    x_init, y = data['x'], data['y']
    condition = np.isnan(y[0])
    x, _ = transforms((x_init, y))

    x = torch.unsqueeze(x, dim=0).float()
    y_hat = model(x.to(device)).to(device)
    y_hat = y_hat.detach().cpu()

    unpad_func = UnPad(TARGET_SIZE)
    y_hat = unpad_func(y_hat[0])[0].numpy()
    y_hat[condition] = np.nan
    
    vmin, vmax = np.nanmin(y), np.nanmax(y)
    levels = np.round(np.linspace(vmin, vmax, 11)).astype(int)


    plot_contour(x_init[1], f'x ({arch} {test_name} config)', GRAPHS_DIR/f'pred/{date}_x_{exp}_{test_name}.png')
    plot_contour(y_hat, f'y_hat ({arch} {test_name} config)', GRAPHS_DIR/f'pred/{date}_yhat_{exp}_{test_name}.png', levels=levels)
    plot_contour(y[0], f'y ({arch} {test_name} config)', GRAPHS_DIR/f'pred/{date}_y_{exp}_{test_name}.png', levels=levels)
    plot_contour(y[0]-y_hat, f'y-y_hat ({arch} {test_name} config)', GRAPHS_DIR/f'pred/{date}_diff_{exp}_{test_name}.png')