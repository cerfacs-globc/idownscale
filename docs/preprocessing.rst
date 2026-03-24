Data Preprocessing
==================

Preprocessing is the foundation of the downscaling pipeline. It converts raw NetCDF data into normalized NumPy samples prepared for neural network training.

Dataset Building
----------------

The primary script is ``bin/preprocessing/build_dataset.py``. It performs:
* Spatial cropping to the experiment domain.
* Conservative interpolation of input variables to the target grid.
* Masking and standardization.
* Packaging into ``.npz`` files.

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

Two scripts are used to compute normalization parameters:

1. ``compute_statistics.py``: Calculates global min, max, mean, and standard deviation across the training set (saved in ``statistics.json``).
2. ``compute_statistics_gamma.py``: Performs pixel-wise Gamma distribution fitting for precipitation variables (saved in ``gamma_params.npz``).
