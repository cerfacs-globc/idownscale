# EGU26 Short Course on using ML for downscaling
**Title:** Using Machine Learning to downscale climate scenarios  
**Conveners:** Christian Pagé, Irida Lazić, Milica Tošić  
**Event:** EGU26 General Assembly, Vienna, Austria & Online | 3–8 May 2026

*In this training it will be shown how Machine Learning can be used in a robust way to downscale coarse climate simulations to very high resolution, suitable for use in climate change adaptation studies. Attendees will learn how to train and apply the ML model, and how to evaluate properly the quality of the results.*

---

## 1. Scientific Strategy: The "Perfect Model" Approach

### Why ERA5 instead of GCM for training?
Global Climate Models (GCMs) generate their own internally consistent weather chronologies. A GCM's "January 1st" will have different storm tracks than the actual historical day. This "chronology mismatch" prevents us from training ML models directly on GCM-Observation pairs—the model cannot learn a mapping if the input (GCM) and target (Observations) show different weather on the same day.

### Strategy
1.  **ERA5 (Reanalysis)**: We use ERA5 as ground truth because it is a high-resolution reconstruction of history.
2.  **Coarsening**: We spatially coarsen ERA5 to simulate the resolution of a GCM.
3.  **Perfect Alignment**: This creates a "Perfect Model" setup where we have aligned input (Coarsened ERA5) and target (High-Res ERA5) for every single day.
4.  **Transfer Learning**: The U-Net learns the spatial physics of downscaling from this perfect pair. This learned mapping is then applied to CMIP6 GCM data.

### Why the U-Net Architecture?
We utilize the **U-Net** topology because it is designed for image-to-image translation. This encoder-decoder architecture captures macro-scale atmospheric patterns (encoder) while using **skip connections** to re-inject high-resolution topographical details directly into the output (decoder). This is essential for downscaling, as local features (mountains, coastlines) remain fixed even as global weather patterns change.

---

## 2. Scientific Constraints & Caveats

### Temperature vs. Precipitation
It is important to note that this specific methodology is highly optimized for **Temperature** (`tas`), but does not apply directly to **Precipitation** (`pr`).

*   **Temperature (Success)**: Temperature is a continuous, Gaussian-like variable with strong spatial correlations and a direct relationship with elevation (lapse rate). Standard Mean Squared Error (MSE) optimization effectively captures these gradients.
*   **Precipitation (Failure)**: Precipitation is sparse (many zeros) and "spiky" (extreme events). It often follows a Gamma distribution rather than Gaussian. Optimizing standard MSE on precipitation results in a "blurry" mean rain field that misses both the location of dry days and the intensity of extreme storms. Specialized loss functions and stochastic architectures are required for robust precipitation downscaling.

### Limitations for Extremes
While the U-Net/MSE approach is excellent for capturing mean states and general spatial patterns, it may **not be the best choice for studying climate extremes**. Optimization via MSE tends to favor the central part of the distribution (smoothing towards the mean) to minimize total error. Consequently, the model may "average out" the most intense events, leading to an underestimation of extreme heatwaves or cold spells. For studies focusing specifically on the tails of the distribution, stochastic methods (like Diffusion Models or GANs) are generally preferred to preserve high-frequency variability.

### Untested Variables
The current setup and hyperparameters have been verified only for surface temperature. Other variables such as **Humidity (huss)**, **Wind Speed (sfcWind)**, and **Sea Level Pressure (psl)** have not been tested or validated within this specific training framework. Users attempting to downscale these variables should proceed with caution and perform independent validation.

---

## 3. Deep Dive: Experiment 5 (EXP5) Focus

Experiment 5 is the primary benchmark for this course. It focuses on high-resolution temperature adaptation studies for the France domain.

*   **Variable**: Daily Surface Maximum Temperature (`tas`).
*   **Region (ROI)**: France `[ -6.0, 10.0, 38.0, 54.0 ]`.
*   **Input Data**: Coarse GCM data (e.g., CNRM-CM6-1) at roughly ~100 km resolution.
*   **Target Data**: E-OBS / ERA5 reanalysis data at ~25 km resolution.
*   **Scientific Goal**: To resolve regional topographical influences (e.g., the Alps, Pyrenees, and Massif Central) on temperature distributions that are typically "smoothed out" in global models.

---

## 3. Technical Configuration & Registry

The pipeline is configured through two primary files in the `iriscc/` directory.

### 3.1 Regional Configuration (`iriscc/settings.py`)
This file defines geographic and structural parameters. It is now **path-neutral**, meaning it resolves data locations relative to the repository root automatically.

*   **`domain`**: A `[min_lon, max_lon, min_lat, max_lat]` list.
*   **`shape`**: Pixel dimensions of the target grid (e.g., `(64, 64)`).
*   **`input_vars` / `target_vars`**: Variables to ingest (elevation, temperature, etc.).

### 3.2 Hyperparameters (`iriscc/hparams.py`)
| Parameter | Default | Purpose |
| :--- | :--- | :--- |
| `learning_rate` | `0.001` | Speed of model weight updates. |
| `batch_size` | `32` | Samples per iteration. |
| `max_epoch` | `60` | Full passes through the training set. |
| `output_norm` | **`True`** | **MANDATORY**: Re-scaling targets to stable gradients (Mean 0, Std 1). |
| `norm_type` | **`standard`** | Specifies **Standard Normalisation** for Gaussian-like variables. |

---

## 4. Environment Setup (Local/Cloud)

Students can set up the environment directly on their laptops or cloud instances. We recommend **Conda** for its robust handling of spatial binary dependencies.

### 4.1 Option A: Conda Setup (Recommended)
```bash
./setup_conda.sh
conda activate idownscale_egu
```
*Note: This script installs `esmf`, `xesmf`, and `sbck` which are essential for climate regridding.*

### 4.2 Option B: Virtual Env Setup
```bash
./setup_venv.sh
source venv_idownscale/bin/activate
```

---

## 5. Execution: Modular Workflow

### 5.1 Manual Step-by-Step
1.  **Build Dataset**: `python bin/preprocessing/build_dataset.py --exp exp5`
2.  **Compute Stats**: `python bin/preprocessing/compute_statistics.py --exp exp5`
3.  **Train**: `python bin/training/train.py`
4.  **Predict**: `python bin/training/predict_loop.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20100101 --enddate 20141231`
5.  **Evaluate**: `python bin/evaluation/compute_test_metrics_day.py --exp exp5 --test-name unet --simu-test gcm_bc`

### 5.2 Automated All-in-One Run
For a quick end-to-end execution of all phases on small test slices:
```bash
# Ensure your environment is active first
./run_generic_workflow.sh
```

### 5.3 HPC & Singularity (Research Mode)
For researchers working on High-Performance Computing (HPC) clusters like **Grace/Calypso**, we provide specialized scripts that utilize **Singularity** containers and **SLURM** job scheduling.
*   **Production Script**: `run_production_pipeline.sh` (or `run_sample_workflow.sh`)
*   **Workflow**: Submit these via `sbatch` to leverage GPU nodes and isolated container environments.

---

## 6. Scientific Lessons: Avoiding Common Pitfalls (Important)

Developing a robust downscaling model reveals several critical numerical pitfalls that apply across climate-AI research.

### 6.1 The 113 K Trap (Target Scale vs. Loss)
*   **The Trap**: Attempting to train directly on absolute Kelvin values (~284 K) using Mean Squared Error (MSE).
*   **Why it Fails**: $|284|^2 \approx 80,000$. The resulting gradients are too massive ($>500$), causing the optimizer to "explode" or stall at a high RMSE (typically around **113 K**).
*   **Solution**: Use `StandardNormalisation` on targets (`output_norm=True`). Bringing inputs/targets into a $\sim 1$ range keeps gradients stable and allows convergence.

### 6.2 The RMSE Gradient Advantage
*   **The Lesson**: Using **RMSE** (the square root of the MSE loss) provides more linear gradients during the final stages of model refinement. This is key in reaching high-fidelity performance versus plateauing early.

### 6.3 The Consistency Constraint
*   **The Trap**: Changing the normalization strategy in the training configuration (`hparams.py`) but failing to update the evaluation script.
*   **Why it Fails**: If the evaluation script is hardcoded to `MinMax`, but the model outputs `StandardNorm` units, it will report an erroneous result (e.g., **102 K RMSE**) even if the model is perfectly trained.
*   **Solution**: Always ensure `train`, `predict`, and `evaluate` use a unified `norm_type` selection logic.

### 6.4 The I/O Performance Wall
*   **The Lesson**: When processing 15+ years of daily data (5,000+ files), simple `glob.glob` lookups inside a loop create a massive metadata bottleneck on HPC file systems.
*   **Success**: Pre-scanning the dataset directory once and using a lookup map reduces evaluation time significantly (e.g., from **30 minutes to 30 seconds**).

---

## 7. Performance Benchmarks

For a successful `exp5` run, evaluate your results against these targets:
*   **RMSE**: Target validated benchmark is **4.15 K**.
*   **Correlation**: Should hit **> 0.73** on the France spatial domain.
*   **Bias Check**: The global cold bias should be eliminated (> -0.5 K) through correct target normalization.

---
**Authors:** **Christian Pagé**, **Irida Lazić**, **Milica Tošić**  
**Credits:** Main codebase and methodology developed by **Zoé Garcia**  
**Repository**: [github.com/cerfacs-globc/idownscale](https://github.com/cerfacs-globc/idownscale)
