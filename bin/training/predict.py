import sys
sys.path.append('.')

import glob
import torch
import numpy as np
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.plotutils import plot_test, plot_contour, plot_map_image
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import GRAPHS_DIR, TARGET_SIZE, RUNS_DIR, DATASET_BC_DIR, CONFIG



def compare_4_subplots(x, y, y_hat, pixel, title, save_dir):
    diff_y = y_hat-y  
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    vmin_y, vmax_y = np.nanmin(y), np.nanmax(y)
    levels_y = np.round(np.linspace(vmin_y, vmax_y, 11)).astype(int)
    levels_diff = np.arange(-5,6,1)

    data = [x, y, y_hat, diff_y]
    subtitles = ["input", "target", "prediction", "prediction - target"]
    cmaps = ["OrRd", "OrRd", "OrRd", "RdBu"]
    levels_list = [levels_y, levels_y, levels_y, levels_diff]

    for i, ax in enumerate(axes.flat):
        if pixel is True:
            if i == 3:
                cs = ax.imshow(np.flip(data[i],axis=0), cmap=cmaps[i],vmin=-5, vmax=5)
            else: 
                cs = ax.imshow(np.flip(data[i],axis=0), cmap=cmaps[i],vmin=vmin_y, vmax=vmax_y)
        else:
            cs = ax.contourf(data[i], cmap=cmaps[i], levels=levels_list[i])
        cbar = plt.colorbar(cs, ax=ax, pad=0.05)
        ax.set_title(subtitles[i], fontsize=12)
        if i == 3:
            cbar.set_label(label='error (K)', size=12)
        else:
            cbar.set_label(label='tas (K)', size=12)

    # Titre général
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    plt.savefig(save_dir)


if __name__=='__main__':
    date = str(sys.argv[1])
    exp = str(sys.argv[2]) # ex : exp 1
    test_name = str(sys.argv[3]) # ex : mask_continents
    cmip6_test = str(sys.argv[4]) # CMIP6, yes or no

    run_dir = RUNS_DIR/f'{exp}/{test_name}/lightning_logs/version_best'
    checkpoint_dir = glob.glob(str(run_dir/f'checkpoints/best-checkpoint*.ckpt'))[0]

    model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
    model.eval()
    hparams = model.hparams['hparams']
    arch = hparams['model']

    transforms = v2.Compose([
                MinMaxNormalisation(hparams['sample_dir'], hparams['output_norm']), 
                LandSeaMask(hparams['mask'], hparams['fill_value']),
                FillMissingValue(hparams['fill_value']),
                Pad(hparams['fill_value'])
                ])
    
    sample_dir = hparams['sample_dir']
    if cmip6_test == 'cmip6' or cmip6_test == 'cmip6_bc':
        test_name = f'{test_name}_{cmip6_test}'
        sample_dir = DATASET_BC_DIR / f'dataset_{exp}_test_{cmip6_test}' # bc or not
    device = 'cpu'

    sample = glob.glob(str(sample_dir/f'sample_{date}.npz'))[0]
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
    x_init = x_init[1]
    x_init[condition] = np.nan

   
    plot_map_image(y_hat,
                   domain = CONFIG['eobs']['domain']['france'],
                   title=f'{date} y_hat {test_name}',
                   save_dir=GRAPHS_DIR/f'pred/{date}_yhat_{exp}_{test_name}.png')

    compare_4_subplots(x_init,
                        y[0], 
                        y_hat, 
                        False,
                        f'{date} {test_name}', 
                        GRAPHS_DIR/f'pred/{date}_subplot_{exp}_{test_name}.png')
    
    
    compare_4_subplots(x_init[10:40,50:80],
                        y[0][10:40,50:80], 
                        y_hat[10:40,50:80], 
                        True,
                        f'{date} {test_name}', 
                        GRAPHS_DIR/f'pred/{date}_subplot_{exp}_{test_name}_local.png')