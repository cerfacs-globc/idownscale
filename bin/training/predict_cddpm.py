"""
Predict and plot results from a trained model.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import glob
import torch
import numpy as np
import argparse
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from iriscc.diffusionutils import generate
from iriscc.lightning_module_ddpm import IRISCCCDDPMLightningModule
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import GRAPHS_DIR, RUNS_DIR, CONFIG, DATASET_BC_DIR


def compare_4_subplots(x, y, y_hat, pixel, title, save_dir):
    diff_y = y_hat-y  
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    vmin_y, vmax_y = np.nanmin(y), np.nanmax(y)
    levels_y = np.round(np.linspace(vmin_y, vmax_y, 11)).astype(int)
    levels_diff = np.arange(-5,6,1)

    data = [x, y, y_hat, diff_y]
    subtitles = ["input", "target", "prediction", "prediction - target"]
    cmaps = ["OrRd", "OrRd", "OrRd", "RdBu"]
    #levels_list = [levels_y, levels_y, levels_y, levels_diff]

    for i, ax in enumerate(axes.flat):
        if pixel is True:
            if i == 3:
                cs = ax.imshow(np.flip(data[i],axis=0), cmap=cmaps[i],vmin=-5, vmax=5)
            else: 
                cs = ax.imshow(np.flip(data[i],axis=0), cmap=cmaps[i],vmin=vmin_y, vmax=vmax_y)
        else:
            #cs = ax.contourf(data[i], cmap=cmaps[i], levels=levels_list[i])
            cs = ax.contourf(data[i], cmap=cmaps[i])

        cbar = plt.colorbar(cs, ax=ax, pad=0.05)
        ax.set_title(subtitles[i], fontsize=12)
        if i == 3:
            cbar.set_label(label='error (K)', size=12)
        else:
            cbar.set_label(label='tas (K)', size=12)

    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    plt.savefig(save_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Predict and plot results")
    parser.add_argument('--date', type=str, help='Date of the sample to predict (format: YYYYMMDD)')
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')   
    parser.add_argument('--test-name', type=str, help='Test name (e.g., mask_continents)')
    parser.add_argument('--simu-test', type=str, help='gcm, gcm_bc, rcm, rcm_bc', default=None)
    args = parser.parse_args()

    run_dir = RUNS_DIR/f'{args.exp}/{args.test_name}/lightning_logs/version_best'
    checkpoint_dir = glob.glob(str(run_dir/f'checkpoints/best-checkpoint*.ckpt'))[0]

    model = IRISCCCDDPMLightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
    model.eval()
    hparams = model.hparams['hparams']
    arch = hparams['model']
    transforms = v2.Compose([
                MinMaxNormalisation(hparams['sample_dir']), 
                LandSeaMask(hparams['mask'], hparams['fill_value']),
                FillMissingValue(hparams['fill_value']),
                Pad(hparams['fill_value'])
                ])
    
    sample_dir = hparams['sample_dir']
    if args.simu_test is not None:
        test_name = f'{args.test_name}_{args.simu_test}'
        sample_dir = DATASET_BC_DIR / f'dataset_{args.exp}_test_{args.simu_test}'
    else : 
        test_name = args.test_name
    device = 'cpu'

    sample = glob.glob(str(sample_dir/f'sample_{args.date}.npz'))[0]
    data = dict(np.load(sample), allow_pickle=True)
    conditioning_image_init, y = data['x'], data['y']

    condition = np.isnan(y[0])
    conditioning_image, _ = transforms((conditioning_image_init, y))

    conditioning_image = torch.unsqueeze(conditioning_image, dim=0).float()
    intermediate_images = generate(model.model,
                                   conditioning_image, 
                                   n_samples=2,
                                   neighbours=False, 
                                   std=1e-1, 
                                   start_t = None, 
                                   clamp=None, 
                                   device='cpu')
    y_hat = intermediate_images[-1][0,...]
    

    unpad_func = UnPad(list(CONFIG[args.exp]['shape']))
    y_hat = unpad_func(torch.Tensor(y_hat))[0].numpy()
    y_hat[condition] = np.nan
    conditioning_image_init = conditioning_image_init[1]
    conditioning_image_init[condition] = np.nan

    
    compare_4_subplots(conditioning_image_init,
                        y[0], 
                        y_hat, 
                        False,
                        f'{args.date} {test_name}', 
                        GRAPHS_DIR/f'pred/{args.date}_subplot_{args.exp}_{test_name}.png')
    
