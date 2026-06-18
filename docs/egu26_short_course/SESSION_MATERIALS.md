# EGU26 Short Course Session Materials

> Release compatibility: this EGU26 short-course material is maintained against `idownscale` release `v1.4.1`. Students who started from the original early-May 2026 material may still see older workflow runner names such as `run_exp5_workflow.py`; see the update notes below before changing commands.

This page gathers the published material for the EGU26 short course on
machine-learning-based climate downscaling.

This is the main entrypoint for students after the session. The practical reading
path is: environment, data layout, upstream inputs, workflow phases, expected
outputs, then the notebook.

## Recommended code checkout

Students who want the same code layout as the maintained course material should
clone the tagged release directly:

```bash
git clone https://github.com/cerfacs-globc/idownscale.git
cd idownscale
git checkout v1.4.1
```

If you already cloned the repository, you can move to the course-compatible
release with:

```bash
git fetch --tags
git checkout v1.4.1
```

## If you started with the early-May 2026 material

There are now two valid paths:

- Keep the historical course flow unchanged:
  stay on the older early-May command set and do not rename workflow commands in
  your notes or scripts. This is the safest option if you already began runs and
  do not want to touch your working setup.
- Move to the maintained repository path:
  update to `v1.4.1` and use the current runner names such as
  `bin/production/run_obs_workflow.py`.

If you choose to update from the older material, the most visible changes are:

- `run_exp5_workflow.py` became `run_obs_workflow.py`
- related wrapper names were aligned to the generic observation-target workflow
- active docs and runbooks now match the maintained code paths and outputs

`v1.4.1` is the better maintained path because it keeps the workflow runner
names, runtime/provenance guidance, active runbooks, and regression checks
aligned in one release.

## Scope versus the full release

The short-course material is narrower than the full `v1.4.1` release. In
particular, the course mainly teaches:

- the `exp5` temperature workflow
- BC-only and BC+ML logic on the daily E-OBS/GCM route
- phase-by-phase validation and interpretation

The full `v1.4.1` release also includes capabilities that are not central to
the short course:

- the generic observation-target workflow runner `run_obs_workflow.py`
- the CERRA observation-target workflow
- frequency-aware workflow controls
- stronger provenance path inventories
- perfect-model BC+CDDPM support and related audit tooling

So the course material is compatible with `v1.4.1`, but it does not try to
teach every workflow that the release now supports.

> **Material update:** The short-course material was updated after the live session
> to clarify the data layout, Mercure packaging, and the optional France-target
> preparation step before Phase 1.

> **Related IRISCC summer school:** For students interested in a longer follow-up
> on the same general topic, IRISCC also plans an on-site summer school,
> *Climate risks data analysis tools and methods*, scheduled for 14-18 September
> 2026 in Chania, Greece.
>
> Training page:
> `https://www.iriscc.eu/training`
>
> Registration page:
> `https://www.iriscc.eu/event/iriscc-onsite-autumn-school-climate-risks-data-analysis-methods/`

## Start here

Recommended reading order:

- [Environment setup](./ENVIRONMENT_SETUP.md)
- [Data setup quickstart](./DATA_SETUP_QUICKSTART.md)
- [Dataset files to provide](./DATASETS_TO_PROVIDE.md)
- [How to fetch upstream data](./HOW_TO_FETCH_UPSTREAM_DATA.md)
- [Local workflow runbook](./LOCAL_WORKFLOW_RUNBOOK.md)
- [Expected outputs by phase](./EXPECTED_PHASE_OUTPUTS.md)
- [Helper scripts](./HELPER_SCRIPTS.md)
- [Workflow phases](./WORKFLOW_PHASES.md)
- [Phase validation](./PHASE_VALIDATION.md)
- [Sample notebook](./egu26_short_course_notebook.ipynb)
- [Presentation PDF](./EGU26_CPAGE_ILAZIC_MTOSIC.pdf)

## Data access

Course material is published through the Mercure space:

- `https://mercure.cerfacs.fr/egu26scml/`

The current top-level Mercure layout is:

- `required/`
- `nice_to_have/`
- `raw_data/`
- `phase_outputs/`
- `egu26_sc_required.tar.gz`
- `egu26_sc_nice_to_have.tar.gz`

This space contains the course companion material and, when useful for offline work,
can also host larger mirrored inputs.

It includes lighter project-specific artifacts such as:

- checkpoint bundle(s)
- selected metrics and plots
- example prediction files
- prepared France-focused files, when useful

It may also include mirrored raw ERA5, E-OBS, and CMIP6 inputs for convenience.
Even when those mirrors are available, the recommended scientific route remains to
fetch the upstream datasets from their official repositories, as described in
[How to fetch upstream data](./HOW_TO_FETCH_UPSTREAM_DATA.md).

## What attendees will find here

The published package supports both a guided short-course session and later
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
- the helper-script, workflow-phase, and phase-validation notes give extra support
  for later offline reruns

## Notes for attendees

- The notebook is intended as a guided example, not as a full production workflow.
- Some phases are inherently long-running because climate training and climate-change
  inference require substantial time periods.
- A trained checkpoint can be reused only when the data and preprocessing setup remain
  compatible with the original training conditions.

## Follow-up contact

For follow-up questions after the short course:

- LinkedIn: `https://www.linkedin.com/in/pagechristian`
