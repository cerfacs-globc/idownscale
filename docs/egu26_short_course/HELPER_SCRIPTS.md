# Helper Scripts For The EGU26 Short Course

This page lists the main helper scripts attendees can use to prepare directories,
crop data, and run the workflow on a laptop, workstation, or supercomputer.

In the notebook, these helper steps should appear near the beginning so users
can bootstrap the workflow easily before moving into the heavier phases.

Before using the helper scripts, users may choose to download and unpack the
published `.tar.gz` files from Mercure as the quickest way to obtain the course assets.

## 1. Directory preparation

The most direct setup helper for the short-course layout is:

- [bin/production/setup_egu26_short_course_tree.sh](../../bin/production/setup_egu26_short_course_tree.sh)

Example:

```bash
bash bin/production/setup_egu26_short_course_tree.sh .
```

This script creates the minimal local tree expected by the short-course notes
and notebook before data are copied or generated.

The runtime layout is controlled by environment variables already used by the repo:

- `IDOWNSCALE_RAW_DIR`
- `IDOWNSCALE_OUTPUT_DIR`
- `IDOWNSCALE_REGRID_WEIGHTS_DIR`
- `IDOWNSCALE_RUNS_DIR`
- `IDOWNSCALE_PREDICTION_DIR`
- `IDOWNSCALE_METRICS_DIR`

Typical local setup:

```bash
export IDOWNSCALE_RAW_DIR=$PWD/rawdata
export IDOWNSCALE_OUTPUT_DIR=$PWD/output
export IDOWNSCALE_REGRID_WEIGHTS_DIR=$IDOWNSCALE_OUTPUT_DIR/regrid_weights
export IDOWNSCALE_RUNS_DIR=$IDOWNSCALE_OUTPUT_DIR/runs
export IDOWNSCALE_PREDICTION_DIR=$IDOWNSCALE_OUTPUT_DIR/prediction
export IDOWNSCALE_METRICS_DIR=$IDOWNSCALE_OUTPUT_DIR/metrics
```

The code creates most output directories automatically. Users mainly need to:

- create or populate `rawdata/`
- choose a writable output root
- keep raw inputs and generated outputs separated

This directory-setup block should be one of the first runnable notebook sections.

For the fastest attendee path after the course, combine this script with:

- [DATA_SETUP_QUICKSTART.md](./DATA_SETUP_QUICKSTART.md)

## 2. France cropping helper

Use [bin/preprocessing/crop_domain.py](../../bin/preprocessing/crop_domain.py) to standardize coordinates and crop a NetCDF file to the `exp5` France domain.

Example:

```bash
python bin/preprocessing/crop_domain.py \
  --input rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc \
  --output rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc \
  --exp exp5 \
  --standardize
```

This same helper can be used for:

- France-cropped E-OBS target temperature
- France-cropped E-OBS elevation
- quick checks on other gridded files before later phases

This cropping step should also be a notebook section of its own, with brief
scientific and technical explanation before the command and output checks after it.

## 3. Main workflow runner

Use [bin/production/run_exp5_workflow.py](../../bin/production/run_exp5_workflow.py) as the main phase-by-phase entrypoint.

Useful examples:

```bash
python bin/production/run_exp5_workflow.py --exp exp5 --steps phase1,stats
python bin/production/run_exp5_workflow.py --exp exp5 --steps phase1,stats,train --test-name unet_all
python bin/production/run_exp5_workflow.py --exp exp5 --steps bc_dataset,bc_apply,raw_dataset
python bin/production/run_exp5_workflow.py --exp exp5 --steps predict_loop,metrics_day,metrics_month,value_metrics --test-name unet_all --simu-test gcm_bc --predict-start-date <STARTDATE> --predict-end-date <ENDDATE> --metrics-start-date <STARTDATE> --metrics-end-date <ENDDATE> --value-start-date <STARTDATE> --value-end-date <ENDDATE>
```

Useful options:

- `--steps` to select only part of the workflow
- `--if-exists skip` to resume work safely
- `--if-exists overwrite` to rebuild selected outputs
- `--phase1-start-date` and `--phase1-end-date` for reduced laptop-scale runs

For the short course, the notebook should prefer running one phase at a time
rather than launching the whole workflow in one block.

## 4. Lower-level phase scripts

Attendees can also call the lower-level scripts directly when they want finer control:

- [bin/preprocessing/build_dataset.py](../../bin/preprocessing/build_dataset.py)
- [bin/preprocessing/compute_statistics.py](../../bin/preprocessing/compute_statistics.py)
- [bin/preprocessing/build_dataset_bc.py](../../bin/preprocessing/build_dataset_bc.py)
- [bin/preprocessing/bias_correction_ibicus.py](../../bin/preprocessing/bias_correction_ibicus.py)
- [bin/training/train.py](../../bin/training/train.py)
- [bin/training/predict_loop.py](../../bin/training/predict_loop.py)

In the short course, the workflow runner should usually be preferred because it
keeps phase ordering and output locations consistent.

## 5. HPC wrapper

For Slurm batch usage, the existing site-specific wrapper can be used as an
example to adapt for the target cluster:

- [bin/production/run_exp5_workflow_grace.sh](../../bin/production/run_exp5_workflow_grace.sh)

That wrapper is useful as a template when attendees move the same workflow from
a local machine to a supercomputer environment.
