Data Preprocessing
==================

Preprocessing is the foundation of the downscaling pipeline. It converts raw NetCDF data into normalized NumPy samples prepared for neural network training.

Dataset Synthesis
-----------------

The pipeline utilizes an **Isolated Volume Protocol** to manage distinct geographical domains:

* **Phase 1 (Reconstruction)**: France domain for historical reanalysis (ERA5).
* **Phase 2 (Bias Correction)**: European domain (29x28 at 1.4°) for GCM synthesis.

The primary script is ``bin/preprocessing/build_dataset.py``. It performs:

* Spatial cropping to the specific experiment domain.
* Archival predictor reconstruction for exp5 via ``ERA5 -> GCM -> E-OBS``.
* Packaging into ``.npz`` daily snapshot volumes.

Certification & Auditing
^^^^^^^^^^^^^^^^^^^^^^^^

Every synthesized volume should be compared against the archival reference when parity matters.
The clean branch keeps those parity audits in ``bin/verification`` and the associated notes
in the repository root.

Resumability & Synthesis
^^^^^^^^^^^^^^^^^^^^^^^^

The cleaned workflow runner supports resumability through ``--if-exists skip`` and explicit rebuilds through ``--if-exists overwrite``.

Statistics & Normalization
--------------------------

Two scripts are used to compute normalization parameters:

1. ``compute_statistics.py``: Calculates global min, max, mean, and standard deviation across the training set (saved in ``statistics.json``).
2. ``compute_statistics_gamma.py``: Performs pixel-wise Gamma distribution fitting for precipitation variables (saved in ``gamma_params.npz``).
