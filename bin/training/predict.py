import sys
sys.path.append('.')

import glob
import torch
import xarray as xr
import numpy as np
from torchvision.transforms import v2
from datetime import datetime
import matplotlib.pyplot as plt

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.plotutils import plot_test, plot_contour
from iriscc.transforms import MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import GRAPHS_DIR, TARGET_SIZE, RUNS_DIR, ERA5_DIR, TARGET_GRID_FILE, DATASET_TEST_6MB_ISAFRAN
from iriscc.datautils import standardize_dims_and_coords, standardize_longitudes, interpolation_target_grid

def get_era5_dataset(date):
    file = glob.glob(str(ERA5_DIR/f'*{date.year}*'))[0]
    ds = xr.open_dataset(file)
    ds = standardize_dims_and_coords(ds)
    ds = standardize_longitudes(ds)
    ds = ds.reindex(lat=ds.lat[::-1])
    ds = ds.sel(lon=slice(-5,10), lat=slice(41.5,51.))
    ds = ds.sel(time=ds.time.dt.date == date.date())
    ds = ds.isel(time=0)
    return ds

def reformat_pred_to_era5(y_hat, ds_era5):
    y_grid = xr.open_dataset(TARGET_GRID_FILE)
    y_ds = xr.Dataset(data_vars=dict(
                            tas=(['y', 'x'], y_hat)),
                    coords=dict(
                            lat=(['y', 'x'], y_grid.lat.values),
                            lon=(['y', 'x'], y_grid.lon.values),
                            y=('y', y_grid.y.values),
                            x=('x', y_grid.x.values)
                            ))
    y_ds = interpolation_target_grid(y_ds, ds_target=ds_era5)
    y_hat = y_ds.tas.values
    y_hat[y_hat == 0.] = np.nan
    return y_hat

def plot_6_subplots(y, y_hat, y_era5, y_hat_reformat, title, save_dir):
    diff_y = y - y_hat
    diff_y_era5 = y_era5 - y_hat_reformat

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmin_y, vmax_y = np.nanmin(y), np.nanmax(y)
    levels_y = np.round(np.linspace(vmin_y, vmax_y, 11)).astype(int)
    levels_diff = np.linspace(-10, 10, 9) 

    data = [y, y_hat, diff_y, y_era5, y_hat_reformat, diff_y_era5]
    subtitles = ["y", "y_hat", "y - y_hat", "y_era5", "y_hat", "y_era5 - y_hat"]
    cmaps = ["OrRd", "OrRd", "RdBu", "OrRd", "OrRd", "RdBu"]
    levels_list = [levels_y, levels_y, levels_diff, levels_y, levels_y, levels_diff]

    for i, ax in enumerate(axes.flat):
        cs = ax.contourf(data[i], cmap=cmaps[i], levels=levels_list[i])
        plt.colorbar(cs, ax=ax, pad=0.05)
        ax.set_title(subtitles[i], fontsize=12)

    # Titre général
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_dir)

if __name__=='__main__':
    date = str(sys.argv[1])
    exp = str(sys.argv[2]) # ex : exp 1
    test_name = str(sys.argv[3]) # ex : mask_continents
    pp_test = str(sys.argv[4]) # Perfect prognosis, yes or no

    run_dir = RUNS_DIR/f'{exp}/{test_name}/lightning_logs/version_best'
    checkpoint_dir = run_dir/'checkpoints/best-checkpoint.ckpt'

    model = IRISCCLightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
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
    if pp_test == 'yes':
        test_name = f'{test_name}_pp'
        sample_dir = DATASET_TEST_6MB_ISAFRAN
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
    ds_era5 = get_era5_dataset(datetime.strptime(date, '%Y%m%d'))
    y_era5 = ds_era5.tas.values

    y_hat_reformat = reformat_pred_to_era5(y_hat, ds_era5)
    condition2 = np.isnan(y_hat_reformat)
    y_era5[condition2] = np.nan

    vmin, vmax = np.nanmin(y), np.nanmax(y)
    levels = np.round(np.linspace(vmin, vmax, 11)).astype(int)

    #plot_contour(x_init[1], f'{date} x ({arch} {test_name} config)', GRAPHS_DIR/f'pred/{date}_x_{exp}_{test_name}.png')
    #plot_contour(y_hat, f'{date} y_hat ({arch} {test_name} config)', GRAPHS_DIR/f'pred/{date}_yhat_{exp}_{test_name}.png', levels=levels)
    #plot_contour(y[0], f'{date} y ({arch} {test_name} config)', GRAPHS_DIR/f'pred/{date}_y_{exp}_{test_name}.png', levels=levels)
    #plot_contour(y_hat-y[0], f'{date} y_hat-y ({arch} {test_name} config)', GRAPHS_DIR/f'pred/{date}_diff_{exp}_{test_name}.png')
    #plot_contour(y_hat_reformat, f'{date} y_hat_reformat ({arch} {test_name} config)', GRAPHS_DIR/f'pred/{date}_yhatreformat_{exp}_{test_name}.png', levels=levels)
    #plot_contour(y_era5, f'{date} y_era5 ({arch} {test_name} config)', GRAPHS_DIR/f'pred/{date}_yera5_{exp}_{test_name}.png', levels=levels)

    plot_6_subplots(y[0], 
                    y_hat, 
                    y_era5, 
                    y_hat_reformat, 
                    f'{date} ({arch} {test_name} config)', 
                    GRAPHS_DIR/f'pred/{date}_subplot_{exp}_{test_name}.png')