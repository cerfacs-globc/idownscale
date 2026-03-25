Data Preprocessing
==================

Preprocessing is the foundation of the downscaling pipeline. It converts raw NetCDF data into normalized NumPy samples prepared for neural network training.

Dataset Building
----------------

The primary script is ``bin/preprocessing/build_dataset.py``. It performs:

* **Domain Filtering**: Subsetting the massive EOBS or SAFRAN source files to your specific experiment area (defined in ``settings.py``).
* **Grid Alignment**: Using ``xesmf`` to perform **conservative interpolation**. This ensures that the energy/mass (e.g., total precipitation or average temperature) is preserved when moving from the source grid to the high-resolution target grid.
* **Normalization Ready**: Extracting the raw values and packaging them into daily ``.npz`` files. Each sample contains the input variables (interpolated GCM/RCM) and the target variable (observations).

.. note::
   This phase is I/O intensive. It reads raw data once and creates thousands of small files to optimize training speed later.

.. code-block:: bash

   python bin/preprocessing/build_dataset.py --exp exp5

Resumability
^^^^^^^^^^^^

All dataset scripts check for existing files. If a job is interrupted, simply restarting it will skip existing dates, saving hours of computation time. To force a full rebuild, use the ``--force True`` flag.

Domain Cropping
---------------

A generic utility ``bin/preprocessing/crop_domain.py`` is available to subset large climate files.

.. code-block:: bash

   python bin/preprocessing/crop_domain.py --input in.nc --output out.nc --exp exp5 --standardize

Statistics & Normalization
--------------------------

2. ``compute_statistics_gamma.py``: Performs pixel-wise Gamma distribution fitting for precipitation variables (saved in ``gamma_params.npz``).

GCM/RCM Interpolation (Phase 2)
------------------------------

Before bias correction, the raw climate model data (GCM or RCM) must be interpolated to the high-resolution target grid of the observations. 

Implementation: ``bin/preprocessing/build_dataset_bc.py``

* **Alignment**: This script takes the large-scale climate files and maps them onto the exact (160x160 or similar) grid used by your observations.
* **Consistency**: It ensures that every GCM/RCM day has a corresponding high-resolution "template" so that the Bias Correction algorithm can operate pixel-by-pixel.

Bias Correction (Phase 3)
-------------------------

Phase 3 uses the ``Ibicus`` library to apply statistical bias correction to the climate model data (GCM or RCM). This step ensures that the model outputs align with the statistical distribution of observations (EOBS or Safran).

Implementation: ``bin/preprocessing/bias_correction_ibicus.py``

The "Silent Initialization" Phase
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the most important aspects of Phase 3 is its initial "silent" period. When you launch this phase, the console will show the start marker and then appear to hang for **15 to 20 minutes**.

**Do not interrupt the process.** During this time, the script is:
1. **Loading Data**: Reading multi-year NetCDF files (1980-2010 for reference, 2015-2100 for future) using ``xarray.open_mfdataset``.
2. **Statistical Fitting**: Using the ``QuantileDeltaMapping`` debiaser to fit historical distributions and compute correction factors. This is a CPU-intensive operation that does not produce any intermediate logs.

Per-Sample Generation
^^^^^^^^^^^^^^^^^^^^^

Once the initial calculations are complete, the script will enter the sample generation loop. You will see logs appearing at a rapid pace:

.. code-block:: text

   [08:35:12] [BC TRAIN] Processing 1980-01-01 (1/7305)
   [08:35:12] [BC TRAIN] Processing 1980-01-02 (2/7305)

At this stage, the script is regridding the debiased data to the target high-resolution grid and saving individual ``.npz`` samples.

Infrastructure and Shape Matching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The script uses a dynamic interpolation strategy to ensure the elevation data (orography) matches the shape of the climate data. If you change the experiment domain or input resolution, Phase 3 will automatically regrid the masks and elevation maps to maintain consistency.
