Data Preprocessing
==================

Preprocessing is the foundation of the downscaling pipeline. It converts raw NetCDF data into normalized NumPy samples prepared for neural network training.

Dataset Synthesis
-----------------

The pipeline utilizes an **Isolated Volume Protocol** to manage distinct geographical domains:

* **Phase 1 (Reconstruction)**: France domain (64x64 at 0.125°) for historical reanalysis (ERA5).
* **Phase 2 (Bias Correction)**: European domain (29x28 at 1.4°) for GCM synthesis.

The primary script is ``bin/preprocessing/build_dataset.py``. It performs:

* Spatial cropping to the specific experiment domain.
* Conservative interpolation of input variables.
* Packaging into ``.npz`` daily snapshot volumes.

Certification & Auditing
^^^^^^^^^^^^^^^^^^^^^^^^

Every synthesized volume is audited by the **Master Census Protocol** (``bin/production/census_p1.py``). This ensures bit-level parity with the archival truth, rejecting any volume with a bias greater than 0.00e+00 K.

Resumability & Synthesis
^^^^^^^^^^^^^^^^^^^^^^^^

All synthesis scripts check for existing daily snapshots. If a job is interrupted, the system automatically fast-forwards to the last completed date, avoiding redundant regridding.

Domain Cropping
---------------

A generic utility ``bin/preprocessing/crop_domain.py`` is available to subset large climate files.

.. code-block:: bash

   python bin/preprocessing/crop_domain.py --input in.nc --output out.nc --exp exp5 --standardize

Statistics & Normalization
--------------------------

Two scripts are used to compute normalization parameters:

1. ``compute_statistics.py``: Calculates global min, max, mean, and standard deviation across the training set (saved in ``statistics.json``).
2. ``compute_statistics_gamma.py``: Performs pixel-wise Gamma distribution fitting for precipitation variables (saved in ``gamma_params.npz``).
