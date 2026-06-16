"""
Predict and save results for a full period by loading a trained model.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append(".")

from pathlib import Path
import xarray as xr
import pandas as pd
import torch
import argparse
import numpy as np
from torchvision.transforms import v2

from iriscc.inference import load_trained_module, predict_tensor
from iriscc.provenance import build_prov_bundle, print_resolved_context, utc_now_iso, write_provjson
from iriscc.runtime_paths import (
    require_match,
    resolve_checkpoint_path,
    resolve_runtime_sample_dir,
    resolve_sample_file_for_timestamp,
    resolve_statistics_dir,
)
from iriscc.transforms import DeMinMaxNormalisation, MinMaxNormalisation, LandSeaMask, Pad, FillMissingValue, UnPad
from iriscc.settings import (
    CONFIG,
    build_time_range,
    get_bc_test_future_dates,
    get_bc_test_hist_dates,
    get_experiment_prediction_frequency,
    get_prediction_output_path,
)
from iriscc.datautils import (remove_countries,
                              Data)


def batched(sequence, batch_size):
    for start in range(0, len(sequence), batch_size):
        yield start, sequence[start : start + batch_size]

def get_target_metadata(exp: str, var: str, dates) -> dict:
    get_data = Data(CONFIG[exp]["domain"])
    if CONFIG[exp].get("target") == "perfect_model":
        source_name = CONFIG[exp].get("perfect_model_target_source") or CONFIG[exp].get("rcm_source")
        if source_name:
            try:
                with get_data._open_source_dataset(source_name, var, date=pd.Timestamp(dates[0]), ssp=CONFIG[exp].get("ssp")) as ds_source:
                    return dict(ds_source[var].attrs) if var in ds_source else {}
            except FileNotFoundError:
                return {}
        return {}
    ds_target = get_data.get_target_dataset(
        target=CONFIG[exp]["target"],
        var=var,
        date=get_bc_test_hist_dates(exp)[-1],
        source_name=CONFIG[exp].get("target_source"),
        skip_domain_crop=bool(CONFIG[exp].get("target_source_pregridded", False)),
    )
    attrs = dict(ds_target[var].attrs) if var in ds_target else {}
    ds_target.close()
    return attrs


def prediction_provenance_attrs(
    exp: str,
    var: str,
    simu_test: str | None,
    test_name: str,
    sample_dir: Path,
    diffusion_num_samples: int,
) -> dict[str, str]:
    cfg = CONFIG[exp]
    attrs = {
        "idownscale_experiment": exp,
        "idownscale_test_name": test_name,
        "idownscale_simu_test": str(simu_test or ""),
        "idownscale_variable": var,
        "idownscale_sample_dir": str(sample_dir),
        "idownscale_diffusion_num_samples": str(diffusion_num_samples),
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


def get_target_format(exp:str, dates, var="tas", sample_dir=None):
    get_data = Data(CONFIG[exp]["domain"])
    attrs = get_target_metadata(exp, var, dates)
    if CONFIG[exp].get("target") == "perfect_model":
        if sample_dir is None:
            sample_dir = CONFIG[exp]["dataset"]
        first_sample = require_match(sample_dir, "sample_*.npz", "sample file", allow_multiple=True)[0]
        first = np.load(first_sample)
        y = first["y"][0]
        ds = xr.Dataset(
            data_vars={var: (["time", "y", "x"], np.empty((len(dates), y.shape[0], y.shape[1])), attrs)},
            coords={"time": dates, "y": np.arange(y.shape[0]), "x": np.arange(y.shape[1])},
        )
        return ds, y
    reference_date = get_bc_test_hist_dates(exp)[-1]
    ds_target = get_data.get_target_dataset(target=CONFIG[exp]["target"],
                                            var=var,
                                            date=reference_date,
                                            source_name=CONFIG[exp].get("target_source"),
                                            skip_domain_crop=bool(CONFIG[exp].get("target_source_pregridded", False)))
    y = ds_target[var].values

    if "x" in ds_target.dims:
        ds = xr.Dataset(
            data_vars={var: (["time", "y", "x"], np.empty((len(dates), y.shape[0], y.shape[1])), attrs)},
            coords={"time" : dates,
                        "y" : ds_target.y.values,
                        "x" : ds_target.x.values})
        if exp == "exp3":
            y = remove_countries(y)
    elif "lon" in ds_target.dims:
        ds = xr.Dataset(
            data_vars={var: (["time", "lat", "lon"], np.empty((len(dates), y.shape[0], y.shape[1])), attrs)},
            coords={"time" : dates,
                        "lat" : ds_target.lat.values,
                        "lon" : ds_target.lon.values})
    return ds, y


if __name__=="__main__":
    start_time = utc_now_iso()
    parser = argparse.ArgumentParser(description="Predict and plot results for full period")
    parser.add_argument("--startdate", type=str, help="Start date (e.g., 20230101)", default=None)
    parser.add_argument("--enddate", type=str, help="End date (e.g., 20230101)", default=None)
    parser.add_argument("--exp", type=str, help="Experiment name (e.g., exp1)")
    parser.add_argument("--test-name", type=str, help="Test name (e.g., mask_continents)")
    parser.add_argument("--simu-test", type=str, help="gcm or gcm_bc, rcm, rcm_bc", default=None)
    parser.add_argument("--var", type=str, default=None, help="Variable to predict. Defaults to the experiment target variable.")
    parser.add_argument("--checkpoint-bundle", type=str, default=None, help="Optional portable checkpoint bundle directory.")
    parser.add_argument("--sample-dir", type=str, default=None, help="Optional explicit sample directory for prediction.")
    parser.add_argument("--batch-size", type=int, default=None, help="Prediction batch size. Defaults to training batch size.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of stochastic diffusion samples to average per prediction.")
    args = parser.parse_args()


    checkpoint_dir = resolve_checkpoint_path(args.exp, args.test_name, args.checkpoint_bundle)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, hparams = load_trained_module(checkpoint_dir, device=device)

    statistics_dir = resolve_statistics_dir(hparams)
    transforms = v2.Compose([
                MinMaxNormalisation(statistics_dir, hparams["output_norm"], hparams.get("output_range", "zero_one")),
                LandSeaMask(hparams["mask"], hparams["fill_value"]),
                FillMissingValue(hparams["fill_value"]),
                Pad(hparams["fill_value"])
                ])
    output_range = hparams.get("output_range", "zero_one")
    denorm = DeMinMaxNormalisation(statistics_dir, hparams["output_norm"], output_range)

    if args.simu_test is not None:
        test_name = f"{args.test_name}_{args.simu_test}"
    else:
        test_name = args.test_name
    sample_dir = resolve_runtime_sample_dir(
        args.exp,
        args.test_name,
        simu_test=args.simu_test,
        hparams=hparams,
        explicit_sample_dir=args.sample_dir,
    )


    startdate = args.startdate or get_bc_test_hist_dates(args.exp)[0].strftime("%Y%m%d")
    enddate = args.enddate or get_bc_test_future_dates(args.exp)[-1].strftime("%Y%m%d")
    var = args.var or CONFIG[args.exp]["target_vars"][0]
    prediction_frequency = get_experiment_prediction_frequency(args.exp)
    dates = build_time_range(startdate, enddate, prediction_frequency)
    ds, y = get_target_format(args.exp, dates=dates, var=var, sample_dir=Path(sample_dir))
    diffusion_num_samples = args.num_samples if hparams.get("model") == "cddpm" else 1
    prediction_path = get_prediction_output_path(
        args.exp,
        args.simu_test,
        var,
        startdate,
        enddate,
        test_name,
        ssp=CONFIG[args.exp].get("ssp"),
    )
    ds.attrs.update(
        prediction_provenance_attrs(
            args.exp,
            var,
            args.simu_test,
            test_name,
            Path(sample_dir),
            diffusion_num_samples,
        )
    )
    target_template = np.expand_dims(y, axis=0)
    batch_size = args.batch_size or int(hparams.get("batch_size", 1)) or 1
    batch_size = max(1, batch_size)
    print_resolved_context(
        script_name="predict_loop.py",
        parameters=vars(args),
        settings={
            "checkpoint_dir": checkpoint_dir,
            "sample_dir": Path(sample_dir),
            "statistics_dir": statistics_dir,
            "batch_size": batch_size,
            "diffusion_num_samples": diffusion_num_samples,
            "output_range": output_range,
            "model": hparams.get("model"),
            "prediction_frequency": prediction_frequency,
        },
        inputs={
            "checkpoint_dir": checkpoint_dir,
            "sample_dir": Path(sample_dir),
            "statistics_json": Path(statistics_dir) / "statistics.json",
        },
        outputs={"prediction_netcdf": prediction_path},
    )
    unpad_func = UnPad(list(target_template.shape[1:]))

    for batch_start, batch_dates in batched(list(dates), batch_size):
        print(f"{batch_dates[0]} to {batch_dates[-1]}")
        xs = []
        masks = []
        target_shapes = []
        for date in batch_dates:
            sample = resolve_sample_file_for_timestamp(sample_dir, date, prediction_frequency)
            data = dict(np.load(sample, allow_pickle=True))

            x = data["x"]
            target_shape = list(data["y"].shape[1:]) if "y" in data else list(target_template.shape[1:])
            x, y_mask = transforms((x, target_template.copy()))
            xs.append(x)
            masks.append(unpad_func(y_mask)[0] == 0)
            target_shapes.append(target_shape)

        if len({tuple(shape) for shape in target_shapes}) != 1:
            raise ValueError(f"Cannot batch variable target shapes: {target_shapes}")

        x_batch = torch.stack(xs, dim=0).float()
        y_hat_batch = predict_tensor(model, x_batch, hparams, device, num_samples=diffusion_num_samples).detach().cpu()

        for offset, condition in enumerate(masks):
            y_hat = unpad_func(y_hat_batch[offset])[0].numpy()
            if hparams["output_norm"]:
                if output_range == "minus_one_one":
                    y_hat = np.clip(y_hat, -1, 1)
                y_hat = denorm((False, np.expand_dims(y_hat, axis=0))).numpy()[0]
            y_hat[condition] = np.nan
            ds[var][batch_start + offset] = y_hat

    ds.to_netcdf(prediction_path)
    prov_path = write_provjson(
        prediction_path.with_suffix(".prov.json"),
        build_prov_bundle(
            script_name="predict_loop.py",
            activity_type="prediction",
            start_time=start_time,
            end_time=utc_now_iso(),
            parameters=vars(args),
            settings={
                "checkpoint_dir": checkpoint_dir,
                "sample_dir": Path(sample_dir),
                "statistics_dir": statistics_dir,
                "batch_size": batch_size,
                "diffusion_num_samples": diffusion_num_samples,
                "output_range": output_range,
                "model": hparams.get("model"),
                "prediction_frequency": prediction_frequency,
            },
            inputs={
                "checkpoint_dir": checkpoint_dir,
                "sample_dir": Path(sample_dir),
                "statistics_json": Path(statistics_dir) / "statistics.json",
            },
            outputs={"prediction_netcdf": prediction_path},
            cwd=Path(__file__).resolve().parents[2],
        ),
    )
    print(f"provenance_provjson={prov_path}", flush=True)
