# Local Workflow Runbook For The EGU26 Short Course

This note assumes the required data are already available locally in the standard
`idownscale` repo layout.

It is meant to help finish the short-course notebook and to provide a clean reference
for the phase-by-phase workflow even before the final data-sharing arrangement is
settled.

Before using this runbook, it is strongly recommended to go through:

- [Environment setup](./ENVIRONMENT_SETUP.md)
- [Data setup quickstart](./DATA_SETUP_QUICKSTART.md)

## 1. Assumptions

This runbook assumes:

- the user is working from the repository root
- the Python environment has already been created
- the main runtime paths have already been set
- the `exp5` case study is used
- the France domain is unchanged
- upstream ERA5 / CMIP6 / E-OBS files are already present locally
- long-running phases remain long-running

The short course should not hide the computational reality of the climate workflow.

## 2. Philosophy

The workflow should be shown as a sequence of explicit phases, each with:

1. one command
2. one or more expected output files
3. one quick validation check
4. one short interpretation

That structure is much easier to teach and to revisit offline than a single large
all-in-one command.

## 3. Optional pre-Phase-1 target preparation

Command:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps prep_phase1
```

Expected outputs:

- `rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`

Optional output if requested directly through the helper:

- `rawdata/eobs/eobs_landseamask_france.nc`

Quick checks:

- confirm target grid is `64x64`
- confirm the France target coordinates match the `exp5` setup

This step is only needed when starting from Europe-scale E-OBS inputs. If the France
target files are already present locally, you can skip it. The France mask is only a
convenience derivative and is not required by the main workflow.

Equivalent direct helper command:

```bash
python bin/preprocessing/prepare_exp5_france_targets.py
```

Optional variants:

```bash
python bin/preprocessing/prepare_exp5_france_targets.py --force
python bin/preprocessing/prepare_exp5_france_targets.py --include-mask
```

## 4. Phase 1: build training samples

Command:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps phase1 \
  --test-name unet_course_demo
```

Expected outputs:

- `datasets/dataset_exp5_30y/sample_YYYYMMDD.npz`

Quick checks:

- open one or two sample files
- verify `x` and `y` shapes
- verify NaN / mask consistency

## 5. Statistics phase

Command:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps stats \
  --test-name unet_course_demo
```

Expected outputs:

- `datasets/dataset_exp5_30y/statistics.json`

Quick checks:

- inspect the JSON keys
- verify the channels match the intended predictor/target setup

## 6. Optional training phase

Command:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps train \
  --test-name unet_course_demo
```

Expected outputs:

- `runs/exp5/unet_course_demo/lightning_logs/version_best/metrics.csv`
- `runs/exp5/unet_course_demo/lightning_logs/version_best/metrics_test_set.csv`
- checkpoint files under `checkpoints/`

Quick checks:

- look at loss evolution
- verify a best checkpoint exists
- inspect test-set metrics

## 7. Prediction phase

Command:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps predict_loop \
  --test-name unet_all \
  --simu-test gcm_bc
```

Expected outputs:

- prediction NetCDF under `prediction/`

Example validated output name:

- `prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_all_gcm_bc.nc`

Quick checks:

- confirm time coverage
- confirm target grid shape
- inspect one time slice

## 8. Metrics phases

Command:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps metrics_day,metrics_month,value_metrics \
  --test-name unet_all \
  --simu-test gcm_bc
```

Expected outputs:

- daily CSV + NPZ
- monthly CSV + NPZ
- VALUE CSV

Quick checks:

- inspect the daily mean CSV
- inspect the monthly mean CSV
- inspect the VALUE summary table

## 9. Plotting phases

Command:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps plot_metrics_day,plot_metrics_month \
  --test-name unet_all \
  --simu-test gcm_bc
```

Expected outputs:

- daily diagnostic figures
- monthly diagnostic figures

Quick checks:

- seasonal RMSE figure
- spatial bias distribution
- spatial RMSE distribution

## 10. Suggested notebook structure

The notebook should alternate:

1. explain the phase
2. show the command
3. list expected output files
4. embed a figure or table snippet
5. explain how to read the result

This allows the notebook to be useful even before users rerun the full workflow.

## 11. Best use of existing validated outputs

While the final executable notebook is still being refined, we can already build a
strong teaching notebook from:

- validated metrics CSVs
- validated plots
- known output filenames
- checkpoint bundle metadata
- training history files

This lets us publish a notebook with meaningful embedded outputs, while real reruns can
continue in parallel later.
