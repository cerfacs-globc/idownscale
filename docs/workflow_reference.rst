Workflow Reference
==================

This page maps the main scripts to the scientific workflow stages. It is meant
to answer the practical question: which script owns which step, and what data
should go in and out?

Observation-Target Workflow
---------------------------

The current master-script entrypoint for the observation-target workflow is
``bin/production/run_obs_workflow.py``. It is the orchestration script used
for the actively maintained observation-target workflows such as ``exp5`` and
``expc``.

The typical step order is:

1. ``prepare_exp5_france_targets.py``
2. ``build_dataset.py``
3. ``compute_statistics.py``
4. ``build_dataset_bc.py``
5. bias correction application script
6. ``build_dataset_pp.py`` for evaluation packaging
7. ``train.py``
8. ``predict_loop.py``
9. metrics and plotting scripts

What Each Step Produces
-----------------------

* target preparation:
  target-grid reference files and supporting static fields such as orography
* phase 1 dataset build:
  packaged ``sample_*.npz`` training dataset
* statistics:
  dataset ``statistics.json`` plus diagnostic histograms
* BC dataset build:
  experiment-specific BC bundles for train/test/future windows
* BC application:
  corrected NetCDF files and packaged BC evaluation samples
* PP/raw dataset packaging:
  packaged evaluation samples at the selected prediction cadence
* training:
  run directory, checkpoints, copied normalization statistics, training
  provenance
* prediction:
  prediction NetCDF and prediction provenance
* metrics:
  daily/monthly/VALUE metrics tables and plots

Perfect-Model Workflow
----------------------

The perfect-model workflow uses a different data contract. The target is not an
observational analysis but a higher-resolution model field, and the evaluation
sample tree is intentionally separated from the training sample tree.

For these runs, pay special attention to:

* ``perfect_model_input_*`` settings
* ``perfect_model_target_*`` settings
* the configured ``evaluation_dataset``
* the prediction/evaluation sample-tree routing

BC-Only, BC+ML, And BC+CDDPM
----------------------------

The same orchestration concepts support three practical modes:

* BC only:
  stop after corrected NetCDF generation and evaluate those outputs directly
* BC+ML:
  package corrected evaluation samples, train a deterministic model such as a
  U-Net, then predict and evaluate
* BC+CDDPM:
  same BC packaging logic, but train and infer with the diffusion model

The important scientific point is that the BC stage is part of the coarse-model
conditioning pipeline, not an afterthought bolted onto only one model family.

Provenance And Validation
-------------------------

The active scripts now print a resolved runtime context and write PROV JSON
files that include:

* key parameters and resolved settings
* important input and output locations
* path inventories for critical files and directories
* timestamps and script names

For long workflows, validation should happen stage by stage:

* confirm expected outputs exist
* confirm date windows and file names match the intended experiment
* confirm prediction/evaluation sample directories match the intended variant
* confirm normalization statistics come from the intended training run

Notebook Examples
-----------------

See :doc:`notebooks` for practical command notebooks covering:

* BC+ML with E-OBS and GCM input
* BC+ML with CERRA and GCM input
* BC-only operation
* BC+CDDPM for perfect-model RCM work
