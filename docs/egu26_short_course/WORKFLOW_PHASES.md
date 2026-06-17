# Workflow Phases For The EGU26 Short Course

> Release compatibility: this EGU26 short-course material is maintained against `idownscale` release `v1.4.0`. If you are using another release, check that workflow runner names, paths, and expected outputs still match that version.

This page summarizes what attendees should do in each workflow phase and which
script to use.

Each phase should be run independently in the notebook. The recommended pattern is:

1. explain the goal of the phase before running anything
2. run only that phase
3. show a small number of outputs such as plots, tables, or array summaries
4. discuss what the outputs mean
5. verify success before moving to the next phase

The notebook should be usable in two ways:

- as a readable technical and scientific walkthrough
- as an executable phase-by-phase workflow that users can rerun later

Some full phases may take 5 to 6 hours. Because of that, the notebook should
make it easy to start from the top, stop after any completed phase, and resume later.

See also:

- [PHASE_VALIDATION.md](./PHASE_VALIDATION.md)

## 1. Prepare directories

Goal:

- define `rawdata/` and output locations
- ensure the `IDOWNSCALE_*` environment variables point to writable paths

Notebook role:

- first practical setup section
- explains the runtime layout and why raw inputs and outputs are separated

Main references:

- [DATA_SETUP_QUICKSTART.md](./DATA_SETUP_QUICKSTART.md)
- [HELPER_SCRIPTS.md](./HELPER_SCRIPTS.md)
- [LOCAL_WORKFLOW_RUNBOOK.md](./LOCAL_WORKFLOW_RUNBOOK.md)
- [EXPECTED_PHASE_OUTPUTS.md](./EXPECTED_PHASE_OUTPUTS.md)
- [PHASE_VALIDATION.md](./PHASE_VALIDATION.md)
- [README.md](../../README.md)

## 2. Pre-Phase-1 France preparation

Goal:

- standardize coordinates if needed
- crop E-OBS target-side files to the France `exp5` domain
- prepare the files used later by the workflow

Notebook role:

- first real data-processing phase
- explains why France cropping is needed scientifically and technically

Main script:

- [bin/preprocessing/crop_domain.py](../../bin/preprocessing/crop_domain.py)

Typical inputs:

- `raw_data/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc`
- `raw_data/eobs/elevation_ens_025deg_reg_v29_0e.nc`

Reference outputs published on Mercure:

- `nice_to_have/eobs_france/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `nice_to_have/eobs_france/elevation_ens_025deg_reg_v29_0e_france.nc`
- `nice_to_have/eobs_france/eobs_landseamask.nc`

What to show in the notebook:

- one map of the cropped target field
- one map of cropped elevation
- a short discussion of the France domain and coordinate checks

## 3. Phase 1 sample generation

Goal:

- build daily `.npz` samples for training, validation, and testing

Notebook role:

- first long workflow phase
- explains how predictors and targets are assembled

Main script:

- [bin/production/run_obs_workflow.py](../../bin/production/run_obs_workflow.py) with `--steps phase1`

Direct lower-level script:

- [bin/preprocessing/build_dataset.py](../../bin/preprocessing/build_dataset.py)

Laptop-friendly reduced example:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps phase1 \
  --phase1-start-date 19850101 \
  --phase1-end-date 19850103
```

What to show in the notebook:

- one table of sample array shapes and value ranges
- one or two example maps extracted from a sample file
- a short discussion of predictor and target channels

## 4. Statistics

Goal:

- compute normalization statistics for the generated samples

Notebook role:

- explains why normalization and summary statistics matter before training
- helps connect the technical preprocessing step to the scientific interpretation of the model inputs and targets

Main script:

- [bin/production/run_obs_workflow.py](../../bin/production/run_obs_workflow.py) with `--steps stats`

Direct lower-level script:

- [bin/preprocessing/compute_statistics.py](../../bin/preprocessing/compute_statistics.py)

What to show in the notebook:

- a compact table extracted from `statistics.json`
- the `hist_y_train.png`, `hist_y_val.png`, and `hist_y_test.png` figures
- a short discussion of whether the sample distributions look reasonable

These statistics and histogram figures can be recomputed live or loaded from the
existing published outputs.

## 5. Bias correction phases

Goal:

- build bias-correction datasets
- apply correction to coarse GCM inputs

Notebook role:

- explains why bias correction is used in this workflow
- shows the difference between raw and corrected coarse inputs
- gives enough context to explain that bias correction does not replace downscaling, but helps reduce mismatch between training and inference inputs

Main workflow steps:

- `bc_dataset`
- `bc_apply`

Direct lower-level scripts:

- [bin/preprocessing/build_dataset_bc.py](../../bin/preprocessing/build_dataset_bc.py)
- [bin/preprocessing/bias_correction_ibicus.py](../../bin/preprocessing/bias_correction_ibicus.py)

What to show in the notebook:

- one table of output file shapes or date ranges
- one comparison plot between raw and corrected coarse inputs
- a short discussion of what changed after correction

## 6. Training or checkpoint reuse

Goal:

- either train a model with the current setup
- or reuse the published checkpoint if the setup is unchanged

Notebook role:

- explains when checkpoint reuse is appropriate
- explains when retraining becomes necessary

Main workflow step:

- `train`

Direct lower-level script:

- [bin/training/train.py](../../bin/training/train.py)

If users change the domain, variables, preprocessing, normalization, or model
setup, they should generally retrain instead of relying on the published checkpoint.

What to show in the notebook:

- a training-loss plot
- a compact metrics table from the training outputs
- a short discussion of whether the run appears stable

## 7. Prediction and evaluation

Goal:

- generate historical or future predictions
- compute daily, monthly, and VALUE-style metrics

Notebook role:

- ties the workflow back to scientific interpretation
- shows the resulting fields, diagnostics, and summary scores
- is the main place where a 1h45 format can spend additional time on discussion and interpretation

Main workflow steps:

- `predict_loop`
- `metrics_day`
- `metrics_month`
- `value_metrics`
- `plot_metrics_day`
- `plot_metrics_month`

Direct lower-level scripts:

- [bin/training/predict_loop.py](../../bin/training/predict_loop.py)
- scripts under [bin/evaluation](../../bin/evaluation/)

What to show in the notebook:

- one prediction map for a selected date
- one daily or monthly summary table
- one or two diagnostic plots
- a short discussion of model behavior and limitations

The end-of-workflow figures should also be shown directly in the notebook, not
only saved to disk. In particular, the notebook should display:

- prediction results themselves
- bias diagnostics
- VALUE metrics summaries
- daily and monthly evaluation plots

When runtime is limited, these end-of-workflow outputs can be taken from the
plots, metrics, and statistics already published on Mercure.

## 8. Environment scaling

The same phase structure should work across environments:

- on a laptop, use short date windows and smaller exploratory runs
- on a workstation, run larger historical segments
- on a supercomputer, run the heavier or full-length phases

The scientific workflow stays the same even when runtime strategy changes.
