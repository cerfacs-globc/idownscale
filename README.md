# IRISCC

![Global projection](/doc/gcm_20700101.png =20%x)  ![Downscaled projection](/doc/unet_20700101.png =20%x)


## Project context

The **IRISCC** project is part of a consortium of European research infrastructures of which CERFACS is a member. Its aim is to study the impacts of risks linked to climate change, by offering services, data and opportunities to the various players involved. The CERFACS mission is to provide high-quality, fine-resolution (around 10 km) climate projection data based on daily data from global climate model (GCM) simulations (around 150 km resolution) for the needs of the project's demonstrators. The variables of interest are surface temperature (K), precipitation (mm) and humidity.

This document provides an overview of the code structure, useful commands for manipulating the project, and the information you need to get started quickly

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
│   ├── dataloaders.py                          # Load rain, valid and test batches
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

Experiment 5 takes E-OBS as reference. A bilinear interpolation is used as baseline.

```bash
python3 bin/preprocessing/build_dataset.py --exp exp5
```
```bash
python3 bin/preprocessing/build_dataset.py --exp exp5 --baseline True
```

To normalize the data, the `compute_statistics.py --exp` script computes statistics for each channel and saves them in a `statistics.json` file in the experiment directory. 
WARNING: If you wish to apply a mask to the inputs, remember to do this step before normalizing.

---

### Training

The IRISCCHyperParameters() class contains all the hyper-parameters required to train neural networks that need to be specified. The command for training the network is :

```bash
python bin/training/train.py
```
The results are saved in the ‘runs’ directory. The progress of metrics can be viewed on Tensorboard with the command :
```bash
tensorboard --logdir='path-to-runs'
```
The path to the best-trained model weights should be renamed ‘{version_best}’ for post-processing.

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