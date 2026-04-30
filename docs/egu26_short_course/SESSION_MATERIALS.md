# EGU26 Short Course Session Materials

This page gathers the online material for the EGU26 short course on
machine-learning-based climate downscaling.

## Course files

- [Dataset files to provide](./DATASETS_TO_PROVIDE.md)
- [Data setup quickstart](./DATA_SETUP_QUICKSTART.md)
- [Environment setup](./ENVIRONMENT_SETUP.md)
- [How to fetch upstream data](./HOW_TO_FETCH_UPSTREAM_DATA.md)
- [Expected outputs by phase](./EXPECTED_PHASE_OUTPUTS.md)
- [Local workflow runbook](./LOCAL_WORKFLOW_RUNBOOK.md)
- [Mercure deploy plan](./MERCURE_DEPLOY_PLAN.md)
- [Sample notebook](./egu26_short_course_notebook.ipynb)
- The presentation PDF will be added here before final publication.

## Data access

Project-specific course artifacts can be shared through the Mercure space:

- `https://mercure.cerfacs.fr/egu26scml/`

This space contains the course companion material and, when useful for offline work,
can also host larger mirrored inputs.

It includes lighter project-specific artifacts such as:

- checkpoint bundle(s)
- selected metrics and plots
- example prediction files
- prepared small France-focused files, when useful

It may also include mirrored raw ERA5, E-OBS, and CMIP6 inputs for convenience.
Even when those mirrors are available, the recommended scientific route remains to
fetch the upstream datasets from their official repositories, as described in
[How to fetch upstream data](./HOW_TO_FETCH_UPSTREAM_DATA.md).

## What attendees will find here

The material is organized to support both a guided short-course session and later
self-study:

- the presentation explains the scientific context and workflow
- the notebook provides a practical phase-by-phase example
- the dataset page distinguishes between shared artifacts and upstream data to fetch
- the environment note explains how to create the Python environment and set the main
  runtime paths
- the quickstart page gives a concrete order for creating directories, unpacking
  Mercure tar files, and placing data in the expected locations
- the upstream-data note explains how to retrieve the large climate inputs
- the runbook and expected-output notes help users interpret each step of the workflow

## Notes for attendees

- The notebook is intended as a guided example, not as a full production workflow.
- Some phases are inherently long-running because climate training and climate-change
  inference require substantial time periods.
- A trained checkpoint can be reused only when the data and preprocessing setup remain
  compatible with the original training conditions.
