"""
Predict and plot results from a trained model.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys

sys.path.append(".")

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.settings import CONFIG, DATASET_BC_DIR, GRAPHS_DIR, RUNS_DIR
from iriscc.transforms import FillMissingValue, LandSeaMask, MinMaxNormalisation, Pad, UnPad


def compare_4_subplots(x, y, y_hat, pixel, title, save_dir):
    diff_y = y_hat - y
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    vmin_y, vmax_y = np.nanmin(y), np.nanmax(y)
    levels_y = np.round(np.linspace(vmin_y, vmax_y, 11)).astype(int)
    levels_diff = np.arange(-5, 6, 1)

    data = [x, y, y_hat, diff_y]
    subtitles = ["input", "target", "prediction", "prediction - target"]
    cmaps = ["OrRd", "OrRd", "OrRd", "RdBu"]
    levels_list = [levels_y, levels_y, levels_y, levels_diff]

    for i, ax in enumerate(axes.flat):
        if pixel is True:
            if i == 3:
                cs = ax.imshow(np.flip(data[i], axis=0), cmap=cmaps[i], vmin=-5, vmax=5)
            else:
                cs = ax.imshow(np.flip(data[i], axis=0), cmap=cmaps[i], vmin=vmin_y, vmax=vmax_y)
        else:
            cs = ax.contourf(data[i], cmap=cmaps[i], levels=levels_list[i])
        cbar = plt.colorbar(cs, ax=ax, pad=0.05)
        ax.set_title(subtitles[i], fontsize=12)
        if i == 3:
            cbar.set_label(label="error (K)", size=12)
        else:
            cbar.set_label(label="tas (K)", size=12)

    # Titre général
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    plt.savefig(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and plot results")
    parser.add_argument("--date", type=str, help="Date of the sample to predict (format: YYYYMMDD)")
    parser.add_argument("--exp", type=str, help="Experiment name (e.g., exp1)")
    parser.add_argument("--test-name", type=str, help="Test name (e.g., mask_continents)")
    parser.add_argument("--simu-test", type=str, help="gcm, gcm_bc, rcm, rcm_bc", default=None)
    args = parser.parse_args()

    # Check for version_best or discover latest numeric version
    from pathlib import Path
    if (Path(RUNS_DIR) / f"{args.exp}/{args.test_name}/lightning_logs/version_best").exists():
        run_dir = Path(RUNS_DIR) / f"{args.exp}/{args.test_name}/lightning_logs/version_best"
    else:
        all_dirs = glob.glob(str(Path(RUNS_DIR) / f"{args.exp}/{args.test_name}/lightning_logs/version_*"))
        # Filter only directories with numeric suffixes
        log_dirs = [d for d in all_dirs if d.split("_")[-1].isdigit()]
        log_dirs = sorted(log_dirs, key=lambda x: int(x.split("_")[-1]))
        if not log_dirs:
             raise FileNotFoundError(f"No numeric version directories found in {Path(RUNS_DIR) / f'{args.exp}/{args.test_name}/lightning_logs/'}")
        run_dir = Path(log_dirs[-1])
    
    checkpoint_dir = glob.glob(str(run_dir / "checkpoints/*.ckpt"))[0]

    model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir, map_location="cpu")
    model.eval()
    hparams = model.hparams["hparams"]
    arch = hparams["model"]
    norm_type = hparams.get("norm_type", "minmax")
    output_norm = hparams.get("output_norm", False)

    from iriscc.transforms import StandardNormalisation, DeStandardNormalisation
    
    if norm_type == "standard":
        norm_func = StandardNormalisation(hparams["sample_dir"], output_norm)
    else:
        norm_func = MinMaxNormalisation(hparams["sample_dir"], output_norm)

    transforms = v2.Compose(
        [
            norm_func,
            LandSeaMask(hparams["mask"], hparams["fill_value"]),
            FillMissingValue(hparams["fill_value"]),
            Pad(hparams["fill_value"]),
        ]
    )

    sample_dir = hparams["sample_dir"]
    if args.simu_test is not None:
        test_name = f"{args.test_name}_{args.simu_test}"
        sample_dir = DATASET_BC_DIR / f"dataset_{args.exp}_test_{args.simu_test}"  # bc or not
    else:
        test_name = args.test_name
    device = "cpu"

    sample = glob.glob(str(sample_dir / f"sample_{args.date}.npz"))[0]
    data = dict(np.load(sample), allow_pickle=True)
    x_init = data["x"]

    # Handle missing target (common in future projections)
    if "y" in data:
        y = data["y"]
        condition = np.isnan(y[0])
        x, _ = transforms((x_init, y))
    else:
        y = None
        condition = np.isnan(x_init[0]) # Use input mask as fallback
        # Create dummy y for transforms if needed
        dummy_y = np.zeros((1, *x_init.shape[1:]))
        x, _ = transforms((x_init, dummy_y))

    x = torch.unsqueeze(x, dim=0).float()
    y_hat = model(x.to(device)).to(device)
    y_hat = y_hat.detach().cpu()

    unpad_func = UnPad(CONFIG[args.exp]["shape"])
    y_hat = unpad_func(y_hat[0]) # Keep (C, H, W)

    if output_norm:
        if norm_type == "standard":
            de_norm = DeStandardNormalisation(hparams["sample_dir"], output_norm)
            y_hat = de_norm((False, y_hat))
        else:
            pass # MinMax handles it differently
    
    y_hat = y_hat[0].numpy() # Now squeeze to (H, W) for plotting/masking
    y_hat[condition] = np.nan
    x_init = x_init[1]
    x_init[condition] = np.nan

    if y is not None:
        compare_4_subplots(x_init, y[0], y_hat, False, f"{args.date} {test_name}", GRAPHS_DIR / f"pred/{args.date}_subplot_{args.exp}_{test_name}.png")
    else:
        # Simple plot for cases without target
        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.imshow(np.flip(y_hat, axis=0), cmap="OrRd")
        plt.colorbar(cs, ax=ax, label="tas (K)")
        ax.set_title(f"{args.date} {test_name} (Prediction)")
        plt.savefig(GRAPHS_DIR / f"pred/{args.date}_prediction_{args.exp}_{test_name}.png")
