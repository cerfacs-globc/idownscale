Release Notes
=============

v1.5.0 - Workflow Generalization, Bias-Correction Extensions, And FAIR Provenance
----------------------------------------------------------------------------------

Summary
~~~~~~~
This release broadens the maintained observation-target workflow well beyond the
``v1.4.1`` alignment update. It generalizes settings and preprocessing for new
experiments, extends the bias-correction and wind-diagnostic path, and upgrades
runtime provenance so production reruns are more auditable and reusable.

Workflow And Settings Generalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Settings layer refactored**: the large runtime settings logic has been
  moved into ``iriscc/settings_base.py`` with ``iriscc/settings.py`` acting as
  the thin experiment-definition layer.
* **Selectable settings modules**: workflows can now be pointed at alternate
  settings modules explicitly, which makes engineer-specific experiment files
  easier to manage without renaming the maintained defaults.
* **Phase-1 temporal chunking**: the observation-target workflow now supports
  sequential phase-1 generation in date chunks for large domains or long
  periods, while keeping the standard sample layout.
* **Statistics split preflight**: workflow guards now validate split anchors
  against the actual generated sample window before statistics are recomputed.

Custom Observation And Static-Field Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Generic custom target-grid support**: the workflow now supports arbitrary
  observation target grids with matching topography and land-sea mask inputs,
  without adding dataset-specific code paths for one-off production cases.
* **Descending-latitude mask fix**: custom observation masks are now handled
  correctly when latitude order is inverted, avoiding geometry/alignment
  failures such as conflicting dimension sizes during preprocessing.
* **Topography helper added and clarified**: the target-topography helper and
  documentation now make it easier to build a dedicated high-resolution static
  field, for example from ETOPO, on the exact target grid used by the
  experiment.

Bias Correction And Workflow Guards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Generic scalar SBCK path**: scalar SBCK handling is now less experiment-
  specific and more reusable across maintained workflows.
* **Paired SBCK MBCn support**: the workflow now supports paired multivariate
  MBCn operation together with downstream hooks for derived wind products.
* **Wind postprocessing and comparison helpers**: derived wind-speed and
  direction products, comparison-suite support, and related diagnostics have
  been added to the maintained path.
* **Sub-daily BC windows corrected**: historical and future BC date windows now
  honor the experiment cadence instead of silently assuming daily sampling.
* **MBCn fit-sample guardrails**: ``sbck_mbcn_max_fit_samples`` is now an
  explicit workflow/settings control, warnings are emitted when capped fits are
  used, and the cap is recorded in provenance because it can affect extremes.
* **BC input validation hardened**: additional checks now validate reference
  and regridded time dimensions, and CERRA multi-file ingestion has been made
  more robust in the maintained workflow.
* **Temporal chunking guidance hardened**: documentation now explains more
  clearly how split dates interact with phase-1 outputs and downstream
  statistics, reducing the risk of NaN normalization failures after manual
  chunked preprocessing.

Provenance, Traceability, And Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Expanded runtime auditing**: provenance sidecars now capture richer command,
  environment, git, dependency, resource, and file-fingerprint metadata.
* **FAIR-oriented provenance model**: provenance now carries a schema version,
  stable run identifiers, parameter-source metadata, model metadata, artifact
  typing, and explicit derivation links between inputs and outputs.
* **Operator documentation refreshed**: getting-started and workflow reference
  pages now describe the stronger provenance behavior and the scientific
  implications of custom masks, topography preparation, and MBCn sampling caps.

v1.4.1 - Workflow Coherence And Documentation Alignment
-------------------------------------------------------

Summary
~~~~~~~
This follow-up update aligns the renamed observation-target workflow runner,
runtime provenance, active operator documentation, and EGU26 teaching material
so the maintained release behaves consistently across code, tests, and user
guides.

Workflow And Runtime Guards
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Runner rename completed**: the maintained observation-target entrypoint is
  now ``bin/production/run_obs_workflow.py`` across active code paths and
  current user-facing documentation.
* **Provenance path inventories**: resolved context and PROV sidecars now
  include inventories for critical paths with existence and file metadata.
* **Workflow resolution regression tests**: prediction, evaluation, and
  perfect-model artifact routing is now locked by targeted tests.
* **Legacy RCM discovery hardened**: old RCM metric scripts now fail loudly on
  missing or ambiguous checkpoint/sample/target-file matches.

Documentation And Training Material
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Operator runbooks refreshed**: active ``doc/`` runbooks now match the
  renamed workflow runner and wrapper scripts.
* **EGU26 compatibility notes**: the short-course material now states that it
  is maintained against ``v1.4.0`` and shows how to check out that exact tag.
* **Course-scope clarification**: the EGU material now explains that it covers
  only a subset of the broader capabilities available in the full release.

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
* **Training integrated in workflow**: ``bin/production/run_obs_workflow.py`` now supports a ``train`` step and training-related overrides.
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
* **Workflow Runner**: ``bin/production/run_obs_workflow.py`` provides a clean observation-target preprocessing entrypoint.
* **Workspace Portability**: environment variables and ``environment.yml`` support laptop and HPC installs.

Documentation
~~~~~~~~~~~~~
* **Sphinx/RTD Support**: RTD material has been restored on the clean branch and updated to reflect the cleaned workflow.
* **Unified Readme**: Updated with portable environment setup and the cleaned observation-target workflow.
