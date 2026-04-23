# Technical report: Statistical downscaling of climate projections from Global Climate Model simulations

**Authors**: Zoé GARCIA (Study engineer), Christian PAGÉ (Research engineer)  
**Date**: November 6, 2024 - August 6, 2025  
**Project**: IRISCC Technical report

---

## 1. Introduction
### 1.1. IRISCC
The IRISCC project is part of a consortium of European research infrastructures. The aim is to study the risk impacts linked to climate change, by offering services, data and opportunities to the various actors involved. The CERFACS mission is to provide high-quality and fine-resolution (around 10 km resolution) daily climate projection from global climate model (GCM) simulations (around 150 km resolution) for the needs of the project’s demonstrators. The variables of interest are mean surface temperature (K), precipitations (mm/day) and humidity.

### 1.2. Statistical downscaling
Statistical downscaling by neural networks is used, taking advantage of the fine resolution of the high resolution (HR) observational data provided by the demonstrators. This will improve the resolution of climate projection data for a chosen scenario, while correcting the bias with respect to observations.

The statistical relationship between large-scale variables and small-scale variables is learned for each data type, in order to recover large-scale information in fine-scale predictions. To achieve this, we will train our neural network on observational data in phase 1 and apply this trained network to simulation data in phase 2. The downscaling function will be calibrated on past data, on the assumption that the function is transferable to the future. Vrac and Ayar (2017) has shown that applying bias correction to large-scale predictors from simulations can improve future predictions.

We use the methodology of Baño Medina et al. (2022) and Soares et al. (2024), consisting of:
1. Training the neural network to downscale ERA5 reanalysis data (interpolated to GCM resolution) to the HR E-OBS observation grid.
2. Applying this trained network to downscale bias-corrected LR simulation data.

The network architectures used are: U-Net (CNN) and SwinUNETR (CNN + Transformers).

---

## 2. Study case: downscale CNRM-CM6-1 projections to E-OBS resolution
The European E-OBS observational dataset (around 25 km) is chosen as the target. Projections from the CNRM-CM6-1 model simulations (around 150 km) are used as GCM data to downscale.

### 2.1. Phase 1: Network calibration
#### 2.1.1. Data pre-processing
ERA5 data are interpolated first to the low resolution (LR) grid to match CNRM-CM6-1 inference format and secondly to the HR grid using conservative interpolation to match network architecture format. The script `build_dataset.py` pre-processes data and creates `.npz` samples: `sample = {'x': np.array, 'y': np.array}`.

Daily samples are stored in a `dataset` directory. Reference topography is added to the inputs always as first channel. Historical statistics (min, max, mean, std) are saved in `statistics.json`.

#### 2.1.2. Training
Training uses PyTorch Lightning. The `train.py` script manages the workflow.
- **Dataloader**: Divides data into training (1985-2004), validation (2005-2009), and test (2010-2014) datasets.
- **Normalization**: `MinMaxNormalisation` scales data between 0 and 1 using `statistics.json`.
- **Loss function**: `MaskedMSELoss` (weighted MSE) for temperature. For precipitation, a gamma-based MAE loss penalizing underestimation (Doury et al., 2024).

**Final Production Benchmark**:
After optimization, a final training is performed over the whole period:
- **Training**: 1980–2009
- **Validation**: 2010–2013
- **Test**: 2014

### 2.2. Phase 2: Network inference
#### 2.2.1. Bias correction
GCM projections are corrected to reduce errors with respect to ERA5 using the **CDF-t** method (ibicus library).
The script `build_dataset_bc.py` creates:
- `bc_train_hist_gcm.npz` (1980-1999)
- `bc_test_hist_gcm.npz` (2000-2014)
- `bc_test_future_gcm.npz` (2015-2100)

The script `bias_correction_ibicus.py` corrects and saves results in the training format in a new dataset directory: `dataset_bc`.

#### 2.2.2. Prediction
Weights from the best checkpoint are loaded to predict HR projections. The script `predict_loop.py` produces a NetCDF file with downscaled projections for a specific period (historical and future).

#### 2.2.3. Evaluation
Evaluation is performed against E-OBS (experimental/historical) and RCM Aladin (future projections) to validate the transferability hypothesis.

---

## 3. Discussion
U-Net is able to extrapolate into the future without underestimating the forced response. SwinUNETR shows lower scores than U-Net for future extrapolation, likely requiring more predictors to leverage 3D spatial information.

---
**CERFACS - Zoé GARCIA**
