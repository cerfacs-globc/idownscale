"""
Predict and save results for a full period by loading a trained model.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import glob
from pathlib import Path
import xarray as xr
import pandas as pd
import torch
import argparse
import numpy as np
from torchvision.transforms import v2

from iriscc.checkpoint_bundle import activate_bundle_contract, resolve_checkpoint_from_bundle
from iriscc.lightning_module import IRISCCLightningModule
from iriscc.transforms import DeMinMaxNormalisation, MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import (RUNS_DIR, 
	                             DATASET_BC_DIR, 
	                             CONFIG,
	                             DATES_BC_TEST_FUTURE,
	                             DATES_BC_TEST_HIST,
	                             get_evaluation_sample_dir,
	                             get_prediction_output_path)
from iriscc.datautils import (remove_countries,
                              Data)

def get_target_metadata(exp: str, var: str, dates) -> dict:
    get_data = Data(CONFIG[exp]['domain'])
    if CONFIG[exp].get('target') == 'perfect_model':
        source_name = CONFIG[exp].get('perfect_model_target_source') or CONFIG[exp].get('rcm_source')
        if source_name:
            with get_data._open_source_dataset(source_name, var, date=pd.Timestamp(dates[0]), ssp=CONFIG[exp].get('ssp')) as ds_source:
                return dict(ds_source[var].attrs) if var in ds_source else {}
        return {}
    ds_target = get_data.get_target_dataset(
        target=CONFIG[exp]['target'],
        var=var,
        date=DATES_BC_TEST_HIST[-1],
        source_name=CONFIG[exp].get('target_source'),
    )
    attrs = dict(ds_target[var].attrs) if var in ds_target else {}
    ds_target.close()
    return attrs


def prediction_provenance_attrs(exp: str, var: str, simu_test: str | None, test_name: str, sample_dir: Path) -> dict[str, str]:
    cfg = CONFIG[exp]
    attrs = {
        "idownscale_experiment": exp,
        "idownscale_test_name": test_name,
        "idownscale_simu_test": str(simu_test or ""),
        "idownscale_variable": var,
        "idownscale_sample_dir": str(sample_dir),
    }
    for key in (
        "target",
        "target_source",
        "perfect_model_input_source",
        "perfect_model_input_resolution",
        "perfect_model_input_grid_source",
        "perfect_model_input_coarse_method",
        "perfect_model_input_target_method",
        "perfect_model_target_source",
        "perfect_model_target_resolution",
        "perfect_model_target_method",
    ):
        if key in cfg:
            attrs[f"idownscale_{key}"] = str(cfg[key])
    return attrs


def get_target_format(exp:str, dates, var='tas', sample_dir=None):
    get_data = Data(CONFIG[exp]['domain'])
    attrs = get_target_metadata(exp, var, dates)
    if CONFIG[exp].get('target') == 'perfect_model':
        if sample_dir is None:
            sample_dir = CONFIG[exp]['dataset']
        first = np.load(next(iter(sorted(sample_dir.glob("sample_*.npz")))))
        y = first["y"][0]
        ds = xr.Dataset(
            data_vars={var: (['time', 'y', 'x'], np.empty((len(dates), y.shape[0], y.shape[1])), attrs)},
            coords={"time": dates, "y": np.arange(y.shape[0]), "x": np.arange(y.shape[1])},
        )
        return ds, y
    reference_date = DATES_BC_TEST_HIST[-1]
    ds_target = get_data.get_target_dataset(target=CONFIG[exp]['target'],
                                            var=var,
                                            date=reference_date,
                                            source_name=CONFIG[exp].get('target_source'))
    y = ds_target[var].values
    
    if 'x' in ds_target.dims:
        ds = xr.Dataset(
            data_vars={var: (['time', 'y', 'x'], np.empty((len(dates), y.shape[0], y.shape[1])), attrs)},
            coords={"time" : dates,
                        "y" : ds_target.y.values,
                        "x" : ds_target.x.values})
        if exp == 'exp3':
            y = remove_countries(y)
    elif 'lon' in ds_target.dims:
        ds = xr.Dataset(
            data_vars={var: (['time', 'lat', 'lon'], np.empty((len(dates), y.shape[0], y.shape[1])), attrs)},
            coords={"time" : dates,
                        "lat" : ds_target.lat.values,
                        "lon" : ds_target.lon.values})
    return ds, y


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Predict and plot results for full period")
    parser.add_argument('--startdate', type=str, help='Start date (e.g., 20230101)', default=DATES_BC_TEST_HIST[0].strftime('%Y%m%d'))
    parser.add_argument('--enddate', type=str, help='End date (e.g., 20230101)', default=DATES_BC_TEST_FUTURE[-1].strftime('%Y%m%d'))
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp1)')
    parser.add_argument('--test-name', type=str, help='Test name (e.g., mask_continents)')
    parser.add_argument('--simu-test', type=str, help='gcm or gcm_bc, rcm, rcm_bc', default=None)
    parser.add_argument('--var', type=str, default=None, help='Variable to predict. Defaults to the experiment target variable.')
    parser.add_argument('--checkpoint-bundle', type=str, default=None, help='Optional portable checkpoint bundle directory.')
    parser.add_argument('--sample-dir', type=str, default=None, help='Optional explicit sample directory for prediction.')
    args = parser.parse_args()


    if args.checkpoint_bundle:
        activate_bundle_contract(args.checkpoint_bundle)
        checkpoint_dir = resolve_checkpoint_from_bundle(args.checkpoint_bundle)
    else:
        run_dir = RUNS_DIR/f'{args.exp}/{args.test_name}/lightning_logs/version_best'
        checkpoint_dir = glob.glob(str(run_dir/f'checkpoints/best-checkpoint*.ckpt'))[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = IRISCCLightningModule.load_from_checkpoint(
        checkpoint_dir,
        map_location=device,
        weights_only=False,
    )
    model.to(device)
    model.eval()
    hparams = model.hparams['hparams']

    transforms = v2.Compose([
                MinMaxNormalisation(hparams['sample_dir'], hparams['output_norm']), 
                LandSeaMask(hparams['mask'], hparams['fill_value']),
                FillMissingValue(hparams['fill_value']),
                Pad(hparams['fill_value'])
                ])
    denorm = DeMinMaxNormalisation(hparams['sample_dir'], hparams['output_norm'])

    sample_dir = hparams['sample_dir']
    if args.simu_test is not None:
        test_name = f'{args.test_name}_{args.simu_test}'
        sample_dir = get_evaluation_sample_dir(args.exp, args.test_name, args.simu_test) or DATASET_BC_DIR / f'dataset_{args.exp}_test_{args.simu_test}' # bc or not
    else:
        test_name = args.test_name
    if args.sample_dir:
        sample_dir = Path(args.sample_dir)

    
    startdate = args.startdate
    enddate = args.enddate
    var = args.var or CONFIG[args.exp]['target_vars'][0]
    dates = pd.date_range(start=startdate, end=enddate, freq='D')
    ds, y = get_target_format(args.exp, dates=dates, var=var, sample_dir=Path(sample_dir))
    ds.attrs.update(prediction_provenance_attrs(args.exp, var, args.simu_test, test_name, Path(sample_dir)))
    target_template = np.expand_dims(y, axis=0)
    
    for i, date in enumerate(dates):
        print(date)
        date_str = date.date().strftime('%Y%m%d')
        sample = glob.glob(str(sample_dir/f'sample_{date_str}.npz'))[0]
        data = dict(np.load(sample, allow_pickle=True))

        x = data['x']
        target_shape = list(data['y'].shape[1:]) if 'y' in data else list(target_template.shape[1:])

        x, y_mask = transforms((x, target_template.copy()))
        condition = y_mask[0] == 0
        x = torch.unsqueeze(x, dim=0).float()
        with torch.no_grad():
            y_hat = model(x.to(device))
        y_hat = y_hat.detach().cpu()

        unpad_func = UnPad(target_shape)
        y_hat = unpad_func(y_hat[0])[0].numpy()
        if hparams['output_norm']:
            y_hat = denorm((False, np.expand_dims(y_hat, axis=0))).numpy()[0]
        y_hat[condition] = np.nan
        ds[var][i] = y_hat

    ds.to_netcdf(
        get_prediction_output_path(
            args.exp,
            args.simu_test,
            var,
            startdate,
            enddate,
            test_name,
            ssp=CONFIG[args.exp].get('ssp'),
        )
    )
    
