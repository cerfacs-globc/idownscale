Release Notes
=============

v1.3.0 - Perfect-Model BC+ML and Provenance Hardening
-----------------------------------------------------

Summary
~~~~~~~
This update corrects the scientific and operational behavior of the
``perfect_model_rcm`` workflow. It fixes silent date-window fallback,
standardizes BC+ML conditioning across the comparison set, and adds explicit
provenance so production reruns can be audited directly from logs and output
directories.

Perfect-Model Workflow
~~~~~~~~~~~~~~~~~~~~~~
* **Correct BC+ML conditioning**: the perfect-model dataset now packages
  elevation, degraded coarse temperature, and bias-corrected coarse temperature
  for ML methods, including CDDPM.
* **Explicit work windows**: ``run_exp5_perfect_model.py`` now accepts
  ``--work-startdate`` and ``--work-enddate`` so prediction, comparison, and
  metrics steps do not silently reuse the historical benchmark window.
* **Model-specific statistics**: denormalization now relies on the matching
  dataset or run ``statistics.json`` instead of silently sharing one file across
  methods.

CDDPM and Evaluation
~~~~~~~~~~~~~~~~~~~~
* **CDDPM parity restored**: the CDDPM perfect-model workflow now uses the
  corrected conditioning path, model-specific normalization, and validated
  prediction outputs.
* **Benchmark outputs regenerated**: historical, future, climate-signal, and
  all-window diagnostics were rerun and refreshed for the corrected workflow.

Provenance and Defaults
~~~~~~~~~~~~~~~~~~~~~~~
* **W3C-style provenance sidecars**: workflow, dataset, training, and
  prediction steps now write ``.prov.json`` files.
* **Resolved-context stdout blocks**: key settings, directories, inputs,
  outputs, and parameters are printed at runtime to expose silent defaults.
* **Wrapper/default cleanup**: path and date-window assumptions were removed
  from the standalone perfect-model launcher and related submit wrappers.

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
