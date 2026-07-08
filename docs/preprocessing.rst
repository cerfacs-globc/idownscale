Data Preprocessing
==================

Preprocessing is the foundation of the downscaling pipeline. It converts raw NetCDF data into normalized NumPy samples prepared for neural network training.

Dataset Synthesis
-----------------

The pipeline utilizes an **Isolated Volume Protocol** to manage distinct geographical domains:

* **Phase 1 (Reconstruction)**: France domain for historical reanalysis (ERA5).
* **Phase 2 (Bias Correction)**: European domain (29x28 at 1.4°) for GCM synthesis.

Source cadence and workflow cadence are now treated separately:

* each upstream source advertises its native frequency in ``SOURCE_CATALOG``
* each experiment resolves a workflow training frequency and prediction
  frequency from its configuration or the target source defaults
* preprocessing can therefore distinguish "native 3-hourly CERRA" from
  "daily samples derived from CERRA"

The primary script is ``bin/preprocessing/build_dataset.py``. It performs:

* Spatial cropping to the specific experiment domain.
* Archival predictor reconstruction for exp5 via ``ERA5 -> GCM -> E-OBS``.
* Packaging into ``.npz`` daily snapshot volumes.

The current sample-packaging layout is still daily for the active workflows.
That is no longer a universal assumption for inference packaging. The active
workflow can now package fixed-step prediction samples at cadences such as
``3h``, ``6h``, or ``12h`` when the upstream source and requested experiment
settings support that conversion safely.

Two boundaries still apply:

* monthly prediction packaging is not implemented in this branch
* prediction packaging will fail loudly if it would need to invent finer-time
  BC or target variability than the available source cadence provides

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

Bias-Correction Methods
-----------------------

The scalar SBCK workflow in ``bin/preprocessing/bias_correction_sbck.py`` is
now variable-generic for scalar fields such as ``tas``, ``pr``, ``psl``, or
``sfcWind``. It no longer assumes temperature-specific variable names when
writing corrected NetCDF files or materializing corrected samples.

Joint multivariate corrections such as wind-vector ``u/v`` with ``MBCn`` are
not part of this scalar path and should be implemented as a dedicated
multivariate workflow.

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
* ``settings``: resolved output roots, target source, grid configuration, and
  resolved training/prediction frequencies
* ``parameters``: explicit dates, channels, and step flags
* ``settings.path_inventory``: resolved input/output paths with existence,
  size, and modification time metadata
