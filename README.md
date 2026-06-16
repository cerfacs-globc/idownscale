# idownscale

## Project context

The **IRISCC** project is part of a consortium of European research infrastructures of which CERFACS is a member. Its aim is to study the impacts of risks linked to climate change, by offering services, data and opportunities to the various players involved. CERFACS tasks in this repository focus on producing high-quality, fine-resolution climate projection products from coarser climate simulations, with daily GCM fields around 150 km as a typical input scale and kilometre-scale regional products as a target use case. The workflow is designed around near-surface temperature today, but should remain adaptable to other climate variables such as precipitation, humidity, wind, or radiation when suitable data and metrics are configured.

This document provides an overview of the code structure, useful commands for manipulating the project, and the information you need to get started quickly

---

## Runtime configuration

The clean branch now centralizes the main path knobs in `iriscc/settings.py`.
The most useful environment variables are:

```bash
export IDOWNSCALE_RAW_DIR=/path/to/rawdata
export IDOWNSCALE_OUTPUT_DIR=/path/to/output
export IDOWNSCALE_GRAPHS_DIR=/path/to/graphs
export IDOWNSCALE_REGRID_WEIGHTS_DIR=/path/to/output/regrid_weights
export IDOWNSCALE_EXP5_ARCHIVE_DATASET_DIR=/path/to/archive/dataset_exp5_30y
export IDOWNSCALE_RUNS_DIR=/path/to/runs
export IDOWNSCALE_PREDICTION_DIR=/path/to/prediction
export IDOWNSCALE_METRICS_DIR=/path/to/metrics
```

Recommended layout on shared filesystems:

```bash
# keep the Git repository separate from heavy runtime data
cd /path/to/idownscale

export IDOWNSCALE_RUNTIME_ROOT=/path/to/idownscale_runtime
export IDOWNSCALE_RAW_DIR=$IDOWNSCALE_RUNTIME_ROOT/rawdata
export IDOWNSCALE_OUTPUT_DIR=$IDOWNSCALE_RUNTIME_ROOT/output
export IDOWNSCALE_GRAPHS_DIR=$IDOWNSCALE_RUNTIME_ROOT/graphs
export IDOWNSCALE_REGRID_WEIGHTS_DIR=$IDOWNSCALE_OUTPUT_DIR/regrid_weights
```

If you do not set them, the repo uses the defaults defined in
`iriscc/settings.py`. In particular:

- `RAW_DIR` uses `repo/rawdata` only if that directory exists
- otherwise `RAW_DIR` falls back to `$IDOWNSCALE_RUNTIME_ROOT/rawdata`
- `OUTPUT_DIR` falls back to `$IDOWNSCALE_RUNTIME_ROOT/output`

For platform-specific environment and operator instructions, keep a local
untracked `iriscc/settings_local.py` or export the variables above. A template is
available in `iriscc/settings_local.py.example`.

Some historical platform notes are kept in `doc/` for project traceability, but
they should not be treated as portable defaults.

Operational note:

- GPU is mainly needed for `train` and `predict_loop`
- preprocessing, bias-correction preparation, metrics, and plotting can run on CPU

---

## Code structure

```
iriscc/
├── bin/                                        # Scripts folder
│   ├── preprocessing/                          # Pre-processing folder
│   │   ├── safran_reformat.py                  # Transform 1D to 2D SAFRAN data 
│   │   ├── aladin_reformat_target.py           # Interpolate ALADIN data to target grid 
│   │   ├── build_dataset.py                    # Build training samples dataset
│   │   ├── build_dataset_bc.py                 # Build bias correction samples dataset
│   │   ├── bias_correction_ibicus.py           # Build corrected training samples dataset
│   │   ├── compute_statistics.py               # Compute min, max, mean and std
│   │   ├── compute_statistics_gamma.py         # Compute alpha and beta
│   ├── training/                               # Training and prediction folder
│   │   ├── train.py                            # Train the network
│   │   ├── predict.py                          # Predict for a date
│   │   ├── predict_loop.py                     # Predict for a time period
│   │   ├── predict_cddpm.py                    # Predict for a date using CDDPM
│   ├── evaluation/                             # Evaluation folder
│   │   ├── compute_test_metrics_day.py         # Compute daily metrics against target
│   │   ├── compute_test_metrics_day_rcm.py     # Compute daily metrics against HR RCM
│   │   ├── compute_test_metrics_month.py       # Compute monthly metrics against target
│   │   ├── compute_test_metrics_month_rcm.py   # Compute monthly metrics against HR RCM
│   │   ├── compare_test_metrics.py             # Plot comparison boxplots
│   │   ├── plot_test_metrics.py                # Plot sptial scores
│   │   ├── plot_histograms.py                  # Plot histograms
│   │   ├── evaluate_futur_trend.py             # Plot maps, histograms and graph 
│   │
├── iriscc/                                     # Usefull functions and class folder
│   ├── models                                  # Neural networks folder
│   │   ├── cddpm.py 
│   │   ├── denoising_unet.py
│   │   ├── miniswinunetr.py
│   │   ├── miniunet.py 
│   │   ├── swin2sr.py
│   │   ├── unet.py
│   ├── dataloaders.py                          # Load train, valid and test batches
│   ├── hparams.py                              # Training configuration
│   ├── settings.py                             # Experience configuration
│   ├── lightning_module.py                     # Training workflow
│   ├── lightning_module_ddpm.py                # Training workflow for CDDPM
│   ├── loss.py                                 # Modified loss class
│   ├── metrics.py                              # Modified metrics class
│   ├── transforms.py                           # Data transformation class
│   ├── plotutils.py                            # Usefull plot functions
│   ├── datautils.py                            # Usefull data functions
│   ├── diffusionutils.py                       # Usefull functions for CDDPM
│   │
├── requirements.txt 
├── README.md 

```

---

## Usefull commands

### Data pre-processing

This command creates training/validation/test samples in the format `sample_{date}.npz`. Conservative interpolation is applied to the inputs to match the size of the output data. Daily samples are stored in a `dataset` directory associated with the experiment. Reference topography is added to the inputs always as first channel.

Experiment 5 takes E-OBS as reference. The archival parity path for predictors uses
`ERA5 -> GCM -> E-OBS` with `conservative_normed`.

```bash
python3 bin/preprocessing/build_dataset.py --exp exp5
```
```bash
python3 bin/preprocessing/build_dataset.py --exp exp5 --baseline True
```

To normalize the data, the `compute_statistics.py --exp` script computes statistics for each channel and saves them in a `statistics.json` file in the experiment directory. 
WARNING: If you wish to apply a mask to the inputs, remember to do this step before normalizing.

---

### Exp5 workflow runner

For the cleaned branch, the recommended entrypoint is:

```bash
python3 bin/production/run_obs_workflow.py --exp exp5
```

By default it runs:
- `phase1`: build Phase 1 samples
- `stats`: compute `statistics.json`
- `bc_dataset`: build Phase 2 coarse bias-correction volumes
- `bc_apply`: apply CDF-t and write corrected coarse files plus inference samples

Optional workflow steps:
- `train`: train a model checkpoint under `runs/<exp>/<test-name>/lightning_logs/version_best`
- `raw_dataset`: build raw test samples for the selected `--simu` source in `dataset_bc/dataset_<exp>_test_<simu>`
- `pp_dataset`: rebuild corrected test samples for the selected `--simu` source in `dataset_bc/dataset_<exp>_test_<simu>_bc`
- `predict_loop`: run long-period inference from an existing trained checkpoint
- `metrics_day`: compute daily evaluation metrics for a prediction product
- `metrics_month`: compute monthly evaluation metrics for a prediction product
- `value_metrics`: compute VALUE-style metrics for a prediction product
- `plot_metrics_day`: generate daily diagnostic figures
- `plot_metrics_month`: generate monthly diagnostic figures

Useful variants:

```bash
python3 bin/production/run_obs_workflow.py --exp exp5 --if-exists skip
python3 bin/production/run_obs_workflow.py --exp exp5 --if-exists overwrite
python3 bin/production/run_obs_workflow.py --exp exp5 --steps phase1,stats --phase1-start-date 19850101 --phase1-end-date 19850131
python3 bin/production/run_obs_workflow.py --exp exp5 --steps phase1,stats,train --test-name unet_all
python3 bin/production/run_obs_workflow.py --exp exp5 --steps bc_dataset,bc_apply,raw_dataset
python3 bin/production/run_obs_workflow.py --exp exp5 --simu rcm --steps bc_dataset,bc_apply,pp_dataset
python3 bin/production/run_obs_workflow.py --exp exp5 --steps predict_loop,value_metrics --test-name unet_all --simu-test gcm_bc --predict-start-date <STARTDATE> --predict-end-date <ENDDATE> --value-start-date <STARTDATE> --value-end-date <ENDDATE>

Date defaults are resolved from `iriscc/settings.py`, but for reproducible reruns it is safer to pass explicit windows whenever you run only part of the workflow.
```

The workflow runner skips fully completed steps by default. With `--if-exists overwrite`,
it removes the relevant outputs for each selected step and rebuilds them.

Checkpoint reuse note:

- a pretrained checkpoint is reusable only when the preprocessing, predictors,
  normalization, and target setup remain compatible
- when the training world changes, the intended route is to rebuild the training
  data and retrain rather than only rerun inference

Prediction note: `predict_loop` expects a `best-checkpoint*.ckpt` under
`$IDOWNSCALE_RUNS_DIR/<exp>/<test-name>/lightning_logs/version_best/checkpoints/`.
The archival `runs/` tree currently preserves logs and hyperparameters but not the checkpoint file itself.

Two lightweight exp5 historical diagnostics are also available once the
workflow products and the `unet_all_gcm_bc` historical prediction exist:

```bash
python bin/evaluation/plot_exp5_historical_5curve.py
python bin/evaluation/plot_exp5_pairwise_distribution_quantiles.py
```

---

### Training

The IRISCCHyperParameters() class contains the main hyper-parameters for training. A reusable CLI entrypoint is:

```bash
python bin/training/train.py --exp exp5 --test-name unet_all
```
Useful overrides include `--model`, `--max-epoch`, `--batch-size`, `--learning-rate`, and `--loss`.

Training now writes directly under:

```bash
$IDOWNSCALE_RUNS_DIR/<exp>/<test-name>/lightning_logs/version_best/
```

The results are saved in the `runs` directory. Where TensorBoard is available,
the progress of metrics can be viewed with:

```bash
tensorboard --logdir='path-to-runs'
```

---
### Bias correction
In the ‘perfect prognosis’ approach employed by [Soares et al. (2024)](https://gmd.copernicus.org/articles/17/229/2024/) and [Vrac and Vaittinada Ayar (2017)](https://journals.ametsoc.org/view/journals/apme/56/1/jamc-d-16-0079.1.xml), the neural network learns the scaling relationship between reanalyses and observations before applying the weights to simulation data. The simulated data are corrected against the reanalyses in pre-processing to reduce model bias.

The CDF-t method [(P.-A. Michelangeli (2009))](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009GL038401) is used here. The data are first pre-processed to create a training set and two data sets (historical and future) to be cleared, one of which will be used to evaluate the method.
```bash
python bin/preprocessing/build_dataset_bc.py --simu gcm --spp ssp585 --var tas
```
The `bin/preprocessing/bias_correction_ibicus.py` script corrects, evaluates and saves the data in the same format used for training the neural network.

```bash
python bin/preprocessing/bias_correction_ibicus.py --exp exp5 --ssp ssp585 --simu gcm --var tas
```


### Prediction
A pre-trained neural network can be used to predict new outputs from inputs it has never seen before. 
A test set is used to compare the prediction with the reference target for a given date. This same test set is used during training to calculate evaluation metrics loaded in Tensorboard. The prediction is obtained by :

```bash
python bin/training/predict.py --date 20121018 --exp exp5 --test-name unet_all --simu-test gcm_bc
```
The following command creates a netCDF file to predict a long period without having to compare with the reference (for the future, for example): 
```bash
python bin/training/predict_loop.py --startdate <STARTDATE> --enddate <ENDDATE> --exp exp5 --test-name unet_all --simu-test gcm_bc
```

Rq: The `--simu-test gcm` option indicates whether the input data is ERA5 (`None`), CNRM-CM6-1 (`gcm`) or corrected CNRM-CM6-1 (`gcm_bc`), as well as for RCM ALADIN (`rcm` and `rcm_bc`). Data are then retrieved from the associated directories.

### Evaluation

The neural network predictions are compared with the reference data for the historical test period.

#### Metrics computing
Daily metrics : 
```bash
python3 bin/evaluation/compute_test_metrics_day.py --startdate <STARTDATE> --enddate <ENDDATE> --exp exp5 --test-name unet
```
Monthly metrics :
```bash
python3 bin/evaluation/compute_test_metrics_month.py --startdate <STARTDATE> --enddate <ENDDATE> --exp exp5 --test-name unet 
```
To calculate metrics in the ‘perfect prognosis’ framework, we need to add the argument `--simu-test gcm_bc`.

To calculate metrics under perfect condition framework (RCM as reference) we use different scripts :

Daily metrics : 
```bash
python3 bin/evaluation/compute_test_metrics_day_rcm.py --startdate <STARTDATE> --enddate <ENDDATE> --exp exp5 --test-name unet
```
Monthly metrics :
```bash
python3 bin/evaluation/compute_test_metrics_month_rcm.py --startdate <STARTDATE> --enddate <ENDDATE> --exp exp5 --test-name unet 
```

#### Score visualization
```bash
python3 bin/evaluation/compare_test_metrics.py --exp exp5 --test-list unet_gcm,unet_gcm_bc --scale monthly --pp pp --simu gcm
```

#### Future trend
The following command creates several figures that compare prediction and GCM (or RCM) future trend.
```bash
python3 bin/evaluation/evaluate_future_trend.py --exp exp5 --ssp ssp585 --simu gcm
```

---

## Supporting docs

- Environment setup notes are documented in the repo docs and engineering notes.
- Additional diagnostics and hygiene notes are maintained as working project documentation.
