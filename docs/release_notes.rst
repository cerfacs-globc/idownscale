Release Notes
=============

v1.2.0 - End-to-End Training, Evaluation, and Packaging
-------------------------------------------------------

Summary
~~~~~~~
This version turns the cleaned ``exp5`` workflow into a practical end-to-end path:
training, long-period inference, daily and monthly evaluation, VALUE-style metrics,
plot generation, and checkpoint packaging are now all supported in a coherent way.

Workflow and Training
~~~~~~~~~~~~~~~~~~~~~
* **Training integrated in workflow**: ``bin/production/run_exp5_workflow.py`` now supports a ``train`` step and training-related overrides.
* **Robust training entrypoint**: ``bin/training/train.py`` now supports reusable CLI arguments for experiment, run name, model, loss, learning rate, batch size, and epoch count.
* **Grace GPU validation**: the training path was validated on Grace GPU with a working documented module and environment combination.

Inference and Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~
* **Prediction path hardened**: long-period inference with ``predict_loop`` is now aligned with the cleaned workflow runner.
* **Daily and monthly metrics validated**: the historical evaluation path produces consistent daily and monthly metrics for retrained checkpoints.
* **VALUE metrics fixed**: invalid-cell masking is now applied consistently, resolving a bug that could corrupt VALUE marginal metrics for retrained outputs.
* **Historical comparison helper**: ``bin/evaluation/compare_exp5_historical_runs.py`` compares archive and candidate runs across CSV and NPZ outputs.

Checkpoint Reuse and Portability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Checkpoint bundle support**: portable checkpoint bundles can now carry a checkpoint together with manifest and setup metadata.
* **Bundle-aware loading**: inference and evaluation can resolve required setup files from a bundle instead of trusting stale historical paths alone.
* **Manifest-first reuse**: checkpoint reuse is now documented as depending on both model weights and a compatible data/preprocessing setup.

Short Course Material
~~~~~~~~~~~~~~~~~~~~~
* **EGU26 short-course pages added**:

  * ``docs/egu26_short_course/SESSION_MATERIALS.md``
  * ``docs/egu26_short_course/SESSION_SUMMARY.md``
  * ``docs/egu26_short_course/DATASETS_TO_PROVIDE.md``

These pages provide a clean public-facing landing point for session materials,
summary text, and dataset publication guidance.

v1.1.0 - Stabilization & Research Readiness
-------------------------------------------

Summary
~~~~~~~
This version marks a critical milestone in the stabilization of the Experiment 5 pipeline. It restores scientific accuracy, introduces a comprehensive validation framework, and optimizes the repository for collaborative research.

Scientific Integrity
~~~~~~~~~~~~~~~~~~~~
* **Temperature Bias Resolution**: Fixed a major regression in Experiment 5 that caused a -180K bias.
* **Current status**: consult the forensic notes and parity logs in the clean branch for the latest verified residuals.
* **Loss Function**: Defaulted back to the stable ``masked_mse`` for EXP5.

Infrastructure & Automation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Workflow Runner**: ``bin/production/run_exp5_workflow.py`` provides a clean exp5 preprocessing entrypoint.
* **Workspace Portability**: environment variables and ``environment.yml`` support laptop and HPC installs.

Documentation
~~~~~~~~~~~~~
* **Sphinx/RTD Support**: RTD material has been restored on the clean branch and updated to reflect the cleaned workflow.
* **Unified Readme**: Updated with portable environment setup and the cleaned exp5 workflow.
