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

France Target Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^

For ``exp5``-style temperature workflows that start from Europe-scale E-OBS inputs,
the repository now also provides a small optional preparation helper:

* ``bin/preprocessing/prepare_exp5_france_targets.py``

This script prepares the France-focused target files used by ``exp5`` from the
Europe-scale E-OBS products:

* France temperature target file
* France elevation file
* optionally a France-focused mask subset

In the workflow runner this appears as the optional step:

* ``prep_phase1``

This step is useful when starting from raw upstream data, but it is not required if
the France-specific target files already exist locally.

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

The statistics file is now treated as a required dataset artifact rather than a
silent shared fallback. Model evaluation and denormalization must resolve a
dataset-specific ``statistics.json`` that matches the actual training samples.

Perfect-Model Sample Layout
---------------------------

The corrected perfect-model temperature workflow uses a bias-correction-plus-ML
design for all ML models, including CDDPM.

For ``perfect_model_rcm`` the packaged samples are:

* ``x[0]``: elevation
* ``x[1]``: degraded coarse predictor
* ``x[2]``: bias-corrected coarse predictor
* ``y[0]``: native-resolution RCM pseudo-truth

This means perfect-model training and evaluation no longer compare BC baselines
against ML models that were denied the BC conditioning field.

Preprocessing Provenance
------------------------

Dataset-building and statistics scripts now write:

* a resolved-context stdout block
* ``provenance_build_dataset.prov.json`` beside the dataset
* ``provenance_statistics.prov.json`` beside the computed statistics outputs

The most useful preprocessing provenance fields are:

* ``inputs``: source files and source roles
* ``outputs``: dataset directory and statistics path
* ``settings``: resolved output roots, target source, and grid configuration
* ``parameters``: explicit dates, channels, and step flags
