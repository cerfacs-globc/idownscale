"""
Predict and plot results from a trained model.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import torch
import numpy as np
import argparse
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from iriscc.inference import load_trained_module, predict_tensor
from iriscc.runtime_paths import (
    resolve_checkpoint_path,
    resolve_runtime_sample_dir,
    resolve_sample_file,
    resolve_statistics_dir,
)
from iriscc.transforms import DeMinMaxNormalisation, MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import GRAPHS_DIR, CONFIG

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

    parser = argparse.ArgumentParser(description="Predict and plot results")
    parser.add_argument('--date', type=str, help='Date of the sample to predict (format: YYYYMMDD)')
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')
    parser.add_argument('--test-name', type=str, help='Test name (e.g., mask_continents)')
    parser.add_argument('--simu-test', type=str, help='gcm, gcm_bc, rcm, rcm_bc', default=None)
    parser.add_argument('--checkpoint-bundle', type=str, default=None, help='Optional portable checkpoint bundle directory.')
    args = parser.parse_args()

    checkpoint_dir = resolve_checkpoint_path(args.exp, args.test_name, args.checkpoint_bundle)

    device = 'cpu'
    model, hparams = load_trained_module(checkpoint_dir, device=device)

    statistics_dir = resolve_statistics_dir(hparams)
    transforms = v2.Compose([
                MinMaxNormalisation(statistics_dir, hparams['output_norm'], hparams.get('output_range', 'zero_one')),
                LandSeaMask(hparams['mask'], hparams['fill_value']),
                FillMissingValue(hparams['fill_value']),
                Pad(hparams['fill_value'])
                ])
    output_range = hparams.get('output_range', 'zero_one')
    denorm = DeMinMaxNormalisation(statistics_dir, hparams['output_norm'], output_range)

    if args.simu_test is not None:
        test_name = f'{args.test_name}_{args.simu_test}'
    else :
        test_name = args.test_name
    sample_dir = resolve_runtime_sample_dir(
        args.exp,
        args.test_name,
        simu_test=args.simu_test,
        hparams=hparams,
    )

    sample = resolve_sample_file(sample_dir, args.date)
    data = dict(np.load(sample), allow_pickle=True)
    x_init, y = data['x'], data['y']

    condition = np.isnan(y[0])
    x, _ = transforms((x_init, y))

    x = torch.unsqueeze(x, dim=0).float()
    y_hat = predict_tensor(model, x, hparams, device).to(device)
    y_hat = y_hat.detach().cpu()

    unpad_func = UnPad(CONFIG[args.exp]['shape'], hparams['fill_value'])
    y_hat = unpad_func(y_hat[0])[0].numpy()
    if hparams['output_norm']:
        if output_range == 'minus_one_one':
            y_hat = np.clip(y_hat, -1, 1)
        y_hat = denorm((False, np.expand_dims(y_hat, axis=0))).numpy()[0]
    y_hat[condition] = np.nan
    x_init = x_init[1]
    x_init[condition] = np.nan

    compare_4_subplots(x_init,
                        y[0],
                        y_hat,
                        False,
                        f'{args.date} {test_name}',
                        GRAPHS_DIR/f'pred/{args.date}_subplot_{args.exp}_{test_name}.png')
