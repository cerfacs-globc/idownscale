Getting Started
===============

The **idownscale** project provides a set of tools for high-resolution climate downscaling using representative deep learning architectures. While the project is optimized for the CERFACS infrastructure, it is designed to be portable to other environments.

The project is designed to be portable across different Linux environments, from local workstations to high-performance compute clusters.

Installation
------------

We recommend using a Conda environment to manage the complex scientific dependencies (like ``xesmf`` and ``ibicus``).

**1. Create the Environment:**

.. code-block:: bash

   conda create -n idownscale python=3.12
   conda activate idownscale

**2. Install Core Dependencies:**

.. code-block:: bash

   # Note: xesmf requires the ESMF library, usually easiest via conda
   conda install -c conda-forge xesmf ibicus
   pip install -r requirements.txt

**3. Configure Paths:**

The project uses a centralized settings file to map your local data directories. Update the following variables in ``iriscc/settings.py``:

.. code-block:: python

   # Example mapping for a local workstation
   DATASET_DIR = Path("/path/to/my/processed_data")
   RAW_DATA_DIR = Path("/path/to/my/raw_netcdf_files")

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
