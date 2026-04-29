# idownscale

<p float="left">
    <img src="/doc/gcm_20700101.png" width="400"/>
    <img src="/doc/unet_20700101.png" width="400"/>
</p>

## Project context

The **IRISCC** project is part of a consortium of European research infrastructures of which CERFACS is a member. Its aim is to study the impacts of risks linked to climate change, by offering services, data and opportunities to the various players involved. The CERFACS mission is to provide high-quality, fine-resolution (around 10 km) climate projection data based on daily data from global climate model (GCM) simulations (around 150 km resolution) for the needs of the project's demonstrators. The variables of interest are surface temperature (K), precipitation (mm) and humidity.

This document provides an overview of the code structure, useful commands for manipulating the project, and the information you need to get started quickly

---

## Runtime configuration

The clean branch now centralizes the main path knobs in [iriscc/settings.py](/scratch/globc/page/idownscale_rerun/iriscc/settings.py).
The most useful environment variables are:

```bash
export IDOWNSCALE_RAW_DIR=/path/to/rawdata
export IDOWNSCALE_OUTPUT_DIR=/path/to/output
export IDOWNSCALE_REGRID_WEIGHTS_DIR=/path/to/output/weights
export IDOWNSCALE_EXP5_ARCHIVE_DATASET_DIR=/path/to/archive/dataset_exp5_30y
export IDOWNSCALE_RUNS_DIR=/path/to/runs
export IDOWNSCALE_PREDICTION_DIR=/path/to/prediction
export IDOWNSCALE_METRICS_DIR=/path/to/metrics
```

If you do not set them, the repo uses the defaults defined in `iriscc/settings.py`.

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

scratch/globc/garcia/
├── datasets/             
├── graphs/              
├── rawdata/             
├── runs/               
├── prediction/          

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
python3 bin/production/run_exp5_workflow.py --exp exp5
```

By default it runs:
- `phase1`: build Phase 1 samples
- `stats`: compute `statistics.json`
- `bc_dataset`: build Phase 2 coarse bias-correction volumes
- `bc_apply`: apply CDF-t and write corrected coarse files plus inference samples

Optional workflow steps:
- `train`: train a model checkpoint under `runs/<exp>/<test-name>/lightning_logs/version_best`
- `raw_dataset`: build raw GCM test samples in `dataset_bc/dataset_<exp>_test_gcm`
- `pp_dataset`: rebuild corrected GCM test samples in `dataset_bc/dataset_<exp>_test_gcm_bc`
- `predict_loop`: run long-period inference from an existing trained checkpoint
- `metrics_day`: compute daily evaluation metrics for a prediction product
- `metrics_month`: compute monthly evaluation metrics for a prediction product
- `value_metrics`: compute VALUE-style metrics for a prediction product
- `plot_metrics_day`: generate daily diagnostic figures
- `plot_metrics_month`: generate monthly diagnostic figures

Useful variants:

```bash
python3 bin/production/run_exp5_workflow.py --exp exp5 --if-exists skip
python3 bin/production/run_exp5_workflow.py --exp exp5 --if-exists overwrite
python3 bin/production/run_exp5_workflow.py --exp exp5 --steps phase1,stats --phase1-start-date 19850101 --phase1-end-date 19850131
python3 bin/production/run_exp5_workflow.py --exp exp5 --steps phase1,stats,train --test-name unet_all
python3 bin/production/run_exp5_workflow.py --exp exp5 --steps bc_dataset,bc_apply,raw_dataset
python3 bin/production/run_exp5_workflow.py --exp exp5 --steps predict_loop,value_metrics --test-name unet_all --simu-test gcm_bc --predict-start-date 20000101 --predict-end-date 20141231
```

The workflow runner skips fully completed steps by default. With `--if-exists overwrite`,
it removes the relevant outputs for each selected step and rebuilds them.

Checkpoint reuse note:

- a pretrained checkpoint is reusable only when the preprocessing, predictors,
  normalization, and target contract remain compatible
- when the training world changes, the intended route is to rebuild the training
  data and retrain rather than only rerun inference

Grace local shell wrapper:

```bash
bash bin/production/run_exp5_workflow_grace.sh --exp exp5 --steps phase1,stats
```

Prediction note: `predict_loop` expects a `best-checkpoint*.ckpt` under
`$IDOWNSCALE_RUNS_DIR/<exp>/<test-name>/lightning_logs/version_best/checkpoints/`.
The archival `runs/` tree currently preserves logs and hyperparameters but not the checkpoint file itself.

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

Grace GPU note:

- known-good module stack:
  - `python/gloenv3.12_arm`
  - `nvidia/cuda/12.4`
- known-good venv:
  - `/scratch/globc/page/idownscale_envs/production_final_v22_312`
- recommended Grace flags:
  - `IDOWNSCALE_FORCE_CSV_LOGGER=1`
  - `IDOWNSCALE_SKIP_TEST_FIGURES=1`

Example Grace training launch:

```bash
sbatch --export=ALL,\
TEST_NAME=unet_smoke,\
STEPS=train,\
IF_EXISTS=overwrite,\
MAX_EPOCH=1,\
IDOWNSCALE_VENV_PATH=/scratch/globc/page/idownscale_envs/production_final_v22_312,\
IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES=tqdm,\
IDOWNSCALE_FORCE_CSV_LOGGER=1,\
IDOWNSCALE_SKIP_TEST_FIGURES=1 \
bin/production/submit_exp5_train_grace.sh
```

For the full engineering note, see `doc/GRACE_TRAINING_ENGINEER_NOTE.md`.

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
python bin/training/predict_loop.py --startdate 20150101 --enddate 21001231 --exp exp5 --test-name unet_all --simu-test gcm_bc
```

Rq: The `--simu-test gcm` option indicates whether the input data is ERA5 (`None`), CNRM-CM6-1 (`gcm`) or corrected CNRM-CM6-1 (`gcm_bc`), as well as for RCM ALADIN (`rcm` and `rcm_bc`). Data are then retrieved from the associated directories.

### Evaluation

The neural network predictions are compared with the reference data for the historical test period.

#### Metrics computing
Daily metrics : 
```bash
python3 bin/evaluation/compute_test_metrics_day.py --startdate 20150101 --enddate 21001231 --exp exp5 --test-name unet
```
Monthly metrics :
```bash
python3 bin/evaluation/compute_test_metrics_month.py --startdate 20150101 --enddate 21001231 --exp exp5 --test-name unet 
```
To calculate metrics in the ‘perfect prognosis’ framework, we need to add the argument `--simu-test gcm_bc`.

To calculate metrics under perfect condition framework (RCM as reference) we use different scripts :

Daily metrics : 
```bash
python3 bin/evaluation/compute_test_metrics_day_rcm.py --startdate 20150101 --enddate 21001231 --exp exp5 --test-name unet
```
Monthly metrics :
```bash
python3 bin/evaluation/compute_test_metrics_month_rcm.py --startdate 20150101 --enddate 21001231 --exp exp5 --test-name unet 
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

- [Environment Setup](/scratch/globc/page/idownscale_rerun/doc/ENVIRONMENT_SETUP.md)
- [Diagnostics Index](/scratch/globc/page/idownscale_rerun/doc/DIAGNOSTICS_INDEX.md)
- [Repo Hygiene Audit](/scratch/globc/page/idownscale_rerun/doc/REPO_HYGIENE_AUDIT.md)
