Getting Started
===============

The **idownscale** project provides a set of tools for high-resolution climate downscaling using representative deep learning architectures. While the project is optimized for the CERFACS infrastructure, it is designed to be portable to other environments.

Installation
------------

The project requires Python 3.11+. We recommend using a Conda environment to manage dependencies.

**General Installation:**

.. code-block:: bash

   conda create -n idownscale_env python=3.11
   conda activate idownscale_env
   pip install -r requirements.txt

**HPC Cluster Example (CERFACS Grace):**

.. code-block:: bash

   module load python/anaconda3.11_arm
   conda activate idownscale_env

Customizing for Your Environment
--------------------------------

To adapt the project to your own cluster or local workstation, you must update the global paths in:

.. code-block:: none

   iriscc/settings.py

Key variables to adjust:

* **DATASET_DIR**: Root directory where generated ``.npz`` samples will be stored.
* **RAW_DATA_DIR**: Directory containing the input NetCDF files (ERA5, GCM, RCM).
* **GRAPHS_DIR**: Output directory for evaluation plots.
* **RUNS_DIR**: Directory for training logs and model weights.

Directory Structure
-------------------

The project expects a structured data environment. By default, it looks for:

* ``datasets/``: Generated samples.
* ``rawdata/``: Input climate data (NetCDF).
* ``graphs/``: Output figures.
* ``runs/``: Training output and checkpoints.

You can modify these mapping in the ``CONFIG`` dictionary inside ``iriscc/settings.py`` for each experiment.
