Package Architecture
====================

The `idownscale` project is structured into two main parts: a core Python package (`iriscc/`) and a set of command-line scripts (`bin/`) that execute the pipeline phases.

Project Structure
-----------------

.. code-block:: text

    .
    ├── iriscc/               # Core library (reusable modules)
    │   ├── registry.py       # Algorithm Registry (Models & Debiasers)
    │   ├── models/           # Neural network architectures
    │   ├── datautils.py      # NetCDF loading, cropping, and formatting
    │   ├── plotutils.py      # Visualization helpers
    │   ├── settings.py       # Global configuration and experiments
    │   ├── hparams.py        # Dynamic hyperparameters (exp-agnostic)
    │   ├── lightning_module.py # PyTorch Lightning wrapper
    │   └── transforms.py     # normalisation & augmentation
    ├── bin/                  # CLI scripts (pipeline execution)
    │   ├── preprocessing/    # Phases 1-3: Dataset building, stats, bias correction
    │   ├── training/         # Phases 4-5: Model training and inference
    │   └── evaluation/       # Phase 6: Metrics and plotting
    └── run_exp5_full.sh      # Master orchestration script

Core Modules (`iriscc/`)
------------------------

Settings and Modularity
~~~~~~~~~~~~~~~~~~~~~~~~~
- **`settings.py`**: The "Single Source of Truth." It defines experiment configurations, including which model to use, which debiaser to apply, and whether to run the AI step (`ai_step: True/False`).
- **`registry.py`**: The central hub for algorithm selection. It dynamically maps configuration strings (e.g., `'cdft'`, `'unet'`) to Python classes from `ibicus`, `SBCK`, or `iriscc.models`.
- **`hparams.py`**: Now experiment-agnostic. It loads its parameters dynamically from the `CONFIG` dictionary in `settings.py` based on the active experiment name.

Data Handling
~~~~~~~~~~~~~
- **`datautils.py`**: Contains the ``Data`` class which handles the complexity of loading NetCDF files from different sources (**EOBS**, **SAFRAN**, **CERRA**, **GCM**, **RCM**). It is fully **variable-agnostic**, supporting temperature, precipitation, wind, and secondary variables through dynamic cleaning and unit normalization.
- **`transforms.py`**: Implements custom TorchVision-style transforms for zero-padding, MinMax normalization, and land-sea masking.

Models and Training
~~~~~~~~~~~~~~~~~~~
- **`models/`**: Pure PyTorch implementations of the architectures.
- **`lightning_module.py`**: Encapsulates the training logic (loss functions, optimizer configuration, validation steps) using PyTorch Lightning to ensure reproducibility and multi-GPU support.

Execution Scripts (`bin/`)
--------------------------

The scripts in `bin/` are designed to be run via the master orchestration script (`run_exp5_full.sh`) or manually. They follow a standard pattern:
1. Load configuration from `iriscc.settings`.
2. check for existence of outputs (resumability).
3. Process data using `iriscc` utilities.
4. Save results to the project-wide directories (`datasets/`, `runs/`, `predictions/`).

Data Flow Overview
------------------

1. **Preprocessing**: Raw NetCDF files are read by `build_dataset.py`, cropped, and saved as `.npz` samples for training.
2. **Bias Correction**: Ibicus algorithm (`bias_correction_ibicus.py`) is applied to GCM/RCM data to align their distributions with observations.
3. **Training**: `train.py` loads the samples, applies `transforms.py`, and optimizes the weights defined in `models/`.
4. **Inference**: `predict_loop.py` uses the trained model to generate high-resolution projections for 100 years of future data.
5. **Evaluation**: `evaluate_futur_trend.py` compares the raw simulation trends with the downscaled model results.
