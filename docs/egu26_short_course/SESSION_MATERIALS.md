# EGU26 Short Course Session Materials

This page gathers the published material for the EGU26 short course on machine-learning-based climate downscaling.

This is the main entrypoint for students after the session. It is intended to
contain the practical path: setup, data access, notebook, workflow phases, and
validation checks.

## Start here

This page is intended to be a GitHub landing page for students. The main reading order is:

- [Data setup quickstart](./DATA_SETUP_QUICKSTART.md)
- [Notebook](./egu26_short_course_notebook.ipynb)
- [Environment setup](./ENVIRONMENT_SETUP.md)
- [Dataset files to provide](./DATASETS_TO_PROVIDE.md)
- [Helper scripts](./HELPER_SCRIPTS.md)
- [Workflow phases](./WORKFLOW_PHASES.md)
- [Phase validation](./PHASE_VALIDATION.md)
- [Local workflow runbook](./LOCAL_WORKFLOW_RUNBOOK.md)
- [Expected phase outputs](./EXPECTED_PHASE_OUTPUTS.md)
- [How to fetch upstream data](./HOW_TO_FETCH_UPSTREAM_DATA.md)
- [Presentation PDF](./EGU26_CPAGE_ILAZIC_MTOSIC.pdf)

## Mercure root

All attendee-facing files are published under:

- `https://mercure.cerfacs.fr/egu26scml/`

The current top-level Mercure layout is:

- `required/`
- `nice_to_have/`
- `raw_data/`
- `phase_outputs/`
- `egu26_sc_required.tar.gz`
- `egu26_sc_nice_to_have.tar.gz`

The two `.tar.gz` files are the easiest bulk-download entrypoint for users who
want to retrieve the course material quickly before running the notebook.

The `idownscale` repository should keep:

- workflow code
- documentation pages
- instructions describing how the published files are produced and used

## Required attendee package

- [Dataset files to provide](./DATASETS_TO_PROVIDE.md)
- [Environment setup](./ENVIRONMENT_SETUP.md)
- [Helper scripts](./HELPER_SCRIPTS.md)
- [Workflow phases](./WORKFLOW_PHASES.md)
- [Phase validation](./PHASE_VALIDATION.md)
- [Notebook](./egu26_short_course_notebook.ipynb)
- `required/notebook/egu26_short_course_notebook.ipynb`
- `required/checkpoint_bundles/exp5_unet_all_bundle/`
- `required/predictions/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_all_gcm_bc.nc`
- `required/metrics/metrics_test_mean_daily_exp5_unet_all_gcm_bc.csv`
- `required/metrics/metrics_test_mean_monthly_exp5_unet_all_gcm_bc.csv`
- `required/metrics/value_metrics_exp5_unet_all.csv`
- `required/metrics/statistics.json`
- selected files from `required/plots/`
- docs mirrored in `required/docs/`

## What attendees will find here

The published package supports both a guided short-course session and later self-study:

- the notebook is meant both to be read as a documented workflow and to be re-executed cell by cell
- the notebook should start with a practical environment-setup section covering Conda, `ESMF`/`esmpy`, `ESMFMKFILE`, and related issues
- the notebook walks through the full workflow, starting with the pre-Phase-1 France cropping step
- the notebook should give users an easy starting sequence: prepare directories, crop France target files, then run each workflow phase independently
- each phase is meant to be run independently, with explanation before and discussion after
- the same notebook can be run on a laptop, workstation, or supercomputer depending on the chosen data volume
- the checkpoint bundle supports pretrained inference, but it is not the only intended path
- the helper-script, workflow-phase, and validation notes show how to prepare directories, how to run each phase, and how to verify success
- the final notebook sections should also display the generated prediction results, bias plots, and VALUE-style summaries
- the notebook can also reuse the plots, metrics, and statistics already published on Mercure when regenerating them during the session would take too long
- the metrics and plots provide validation examples without rerunning everything
- the `.tar.gz` packages provide a simple way to download the required or supplementary published assets in one step
- the `raw_data/`, `nice_to_have/`, and `phase_outputs/` trees expose the larger companion data products
- the tarballs provide bundled downloads of the required and supplementary packages

## Notes for attendees

- The notebook is intended to cover all workflow phases, not only pretrained inference.
- The notebook should combine detailed technical explanation and scientific explanation with runnable commands and inspection cells.
- The first workflow step for many attendees will be the France-focused target preparation and cropping step before Phase 1 sample generation.
- Reduced temporal windows can be used on a laptop for portability, while the same commands can be scaled up on a workstation or HPC system for fuller runs.
- Some phases may take several hours in a full run, so it should always be possible to stop after one phase, inspect outputs, and resume later.
- A trained checkpoint can be reused only when the data and preprocessing setup remain compatible with the training configuration.
- If users change the domain, variables, preprocessing, normalization, architecture, or other core configuration choices, they should expect to retrain rather than rely on the published checkpoint.
- The `required/` tree is the primary attendee entrypoint.
- The `nice_to_have/` tree contains optional companion products such as the `exp5_swinunet_all_bundle/` bundle and additional diagnostics.

## Follow-up contact

For follow-up questions after the short course:

- LinkedIn: `https://fr.linkedin.com/in/pagechristian`
