Workflow Management
===================

The **idownscale** pipeline is managed through an automated master script ``run_exp5_full.sh``. This script coordinates the execution of multiple preprocessing, training, and evaluation phases.

Running the Pipeline
--------------------

The pipeline is designed to be run as a workload manager (Slurm) batch script, but it can also be executed interactively.

**Interactive Execution:**

.. code-block:: bash

   ./run_exp5_full.sh

**Workload Manager (Slurm):**

.. code-block:: bash

   sbatch run_exp5_full.sh

Execution Controls
------------------

The script supports environment variables for granular control:

* **START_PHASE** & **STOP_PHASE**: Run a specific range of phases (e.g., ``START_PHASE=4 STOP_PHASE=4`` for Training only).
* **FORCE=1**: Bypasses the marker check (``.markers/`` directory) in the bash script. Scripts will still **resume** from the last available date if possible.
* **REGENERATE=1**: Forces a full data regeneration by passing the ``--force`` flag to individual Python scripts.

Monitoring & Logging
--------------------

When monitoring long-running jobs, look for these healthy patterns:

Fast-Forwarding (Resumability)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If internal samples already exist, the scripts will skip the heavy interpolation work. Logs will show "Processing date..." at a very high rate. This is the expected behavior for resumed jobs.

Silent Computation Phases
^^^^^^^^^^^^^^^^^^^^^^^^^^

Some steps are computationally intensive and natively silent:

* ``compute_statistics_gamma.py``: Fits Gamma distributions for every pixel.
* ``bias_correction_ibicus.py``: The debiasing loop can be silent for long periods.

Customizing Experiments
-----------------------

The modular framework allows you to define new experiments or sensitivity tests by adding entries to the ``CONFIG`` dictionary in ``iriscc/settings.py``.

Defining a New Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~

A typical experiment configuration includes:

.. code-block:: python

    'exp_name': {
        'target': 'cerra',          # Reference dataset (Observation/Ground Truth)
        'input_source': 'era5',     # Input reanalysis (e.g., 'era5', 'cerra')
        'target_vars': ['pr'],      # Any climate variable (tas, pr, uas, etc.)
        'input_vars': ['elevation', 'pr'], # Predictors
        'debiaser': 'cdft',         # method (e.g., 'cdft', 'quantile_delta_mapping')
        'model': 'none',            # 'unet', 'swin2sr', or 'none' for BC-only
        'ai_step': False,           # Enable/Disable training & inference
        'remove_countries': True,   # Optional: Mask limitrophe countries (SAFRAN style)
        'dataset': DATASET_DIR / 'dataset_exp_name',
    }

Example Configurations
~~~~~~~~~~~~~~~~~~~~~~

Below are several examples of how to define experiments for common use cases.

Example 0: Standard Experiment (Exp5)
--------------------------------------
This is the baseline configuration for downscaling temperature over France using ERA5 and EOBS.

.. code-block:: python

    'exp5': {
        'target': 'eobs',
        'input_source': 'era5',
        'target_vars': ['tas'],
        'input_vars': ['elevation', 'tas'],
        'ai_step': True,
        'model': 'unet',
        'dataset': DATASET_EXP5_30Y_DIR,
    }

Example 1: CERRA as High-Resolution Target
------------------------------------------
In this case, we use CERRA as the "ground truth" for training the AI model to downscale ERA5.

.. code-block:: python

    'exp_cerra_target': {
        'target': 'cerra',
        'input_source': 'era5',
        'target_vars': ['tas'],
        'input_vars': ['elevation', 'tas'],
        'ai_step': True,
        'model': 'unet',
        'dataset': DATASET_DIR / 'dataset_cerra_target',
    }

Example 2: CERRA as Reanalysis Input
------------------------------------
Here, we use CERRA to downscale a GCM/RCM, with EOBS as the reference.

.. code-block:: python

    'exp_cerra_input': {
        'target': 'eobs',
        'input_source': 'cerra',
        'target_vars': ['tas'],
        'input_vars': ['elevation', 'tas'],
        'ai_step': True,
        'model': 'unet',
        'dataset': DATASET_DIR / 'dataset_cerra_input',
    }

Example 3: Precipitation (Bias Correction Only)
-----------------------------------------------
For precipitation, we often use only the Bias Correction step to ensure multivariate coherence.

.. code-block:: python

    'exp_pr_bc_only': {
        'target': 'safran',
        'target_vars': ['pr'],
        'input_vars': ['elevation', 'pr'],
        'ai_step': False,           # Disables AI training and inference
        'debiaser': 'cdft',         # Swappable via registry
        'remove_countries': True,   # Masks limitrophe countries
        'dataset': DATASET_DIR / 'dataset_pr_bc',
    }

Example 4: Multivariate AI Predictors
-------------------------------------
Using multiple physical variables (humidity, pressure) to predict a target variable.

.. code-block:: python

    'exp_multivariate': {
        'target': 'eobs',
        'target_vars': ['pr'],
        'input_vars': ['elevation', 'huss', 'psl', 'tas'],
        'channels': ['elevation', 'huss input', 'psl input', 'tas input', 'pr target'],
        'ai_step': True,
        'model': 'swin2sr',
        'dataset': DATASET_DIR / 'dataset_multivariate',
    }


Running a Custom Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  **Preprocessing**: Call scripts with ``--exp exp_name``.
2.  **Training**: Use the ``--exp`` flag: ``python bin/training/train.py --exp exp_name``.
3.  **Inference**: Use the ``--exp`` flag in ``predict_loop.py``.

If ``ai_step`` is ``False``, Phase 4 (Training) and Phase 5 (Inference) are skipped, and Phase 6 (Evaluation) uses the bias-corrected NetCDF files as the authoritative source.

Customizing for Your Cluster (Example)
--------------------------------------

If you are using a different cluster or partition, update the ``#SBATCH`` headers in the script. For example, on a cluster with a dedicated CPU partition:

.. code-block:: bash

   # Example: Using the Grace partition (CPU-only)
   #SBATCH -p grace        
   #SBATCH --gres=gpu:0
