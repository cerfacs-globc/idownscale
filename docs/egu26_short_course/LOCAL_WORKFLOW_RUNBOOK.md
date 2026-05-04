# Local Workflow Runbook For The EGU26 Short Course

This note assumes the required data are already available locally in the
standard `idownscale` repo layout.

Read this after:

- [ENVIRONMENT_SETUP.md](./ENVIRONMENT_SETUP.md)
- [DATA_SETUP_QUICKSTART.md](./DATA_SETUP_QUICKSTART.md)

## 1. Assumptions

This runbook assumes:

- you work from the repository root
- the Python environment is already set up
- the main runtime paths are writable
- the `exp5` case study is used
- upstream ERA5, CMIP6, and E-OBS files are already present locally

## 2. Teaching rhythm

Each phase should be shown with:

1. one command
2. one or more expected outputs
3. one quick validation check
4. one short interpretation

## 3. Optional France target preparation

If the France-cropped E-OBS targets are not already present, crop them first
with [bin/preprocessing/crop_domain.py](/Users/page/src/idownscale/bin/preprocessing/crop_domain.py).

Temperature:

```bash
python bin/preprocessing/crop_domain.py \
  --input rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc \
  --output rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc \
  --exp exp5 \
  --standardize
```

Elevation:

```bash
python bin/preprocessing/crop_domain.py \
  --input rawdata/eobs/elevation_ens_025deg_reg_v29_0e.nc \
  --output rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc \
  --exp exp5 \
  --standardize
```

Quick checks:

- confirm the files exist under `rawdata/eobs/`
- confirm the target grid matches the France `exp5` setup

## 4. Phase 1: build training samples

Command:

```bash
python bin/production/run_exp5_workflow.py \
  --exp exp5 \
  --steps phase1 \
  --test-name unet_course_demo
```

Expected outputs:

- `idownscale_output/datasets/dataset_exp5_30y/sample_YYYYMMDD.npz`

Quick checks:

- open one or two sample files
- verify `x` and `y` shapes
- verify NaN and mask behavior

## 5. Statistics phase

Command:

```bash
python bin/production/run_exp5_workflow.py \
  --exp exp5 \
  --steps stats \
  --test-name unet_course_demo
```

Expected outputs:

- `idownscale_output/datasets/dataset_exp5_30y/statistics.json`
- `idownscale_output/datasets/dataset_exp5_30y/hist_y_train.png`
- `idownscale_output/datasets/dataset_exp5_30y/hist_y_val.png`
- `idownscale_output/datasets/dataset_exp5_30y/hist_y_test.png`

Quick checks:

- inspect the JSON keys
- verify the channels match the intended setup
- show the histogram figures

## 6. Bias-correction phases

Command:

```bash
python bin/production/run_exp5_workflow.py \
  --exp exp5 \
  --steps bc_dataset,bc_apply
```

Expected outputs:

- bias-correction `.npz` files under the bias-correction dataset directory
- corrected GCM NetCDF files

Quick checks:

- verify the corrected files exist
- inspect one historical and one future file
- compare raw and corrected coarse inputs

## 7. Optional training phase

Command:

```bash
python bin/production/run_exp5_workflow.py \
  --exp exp5 \
  --steps train \
  --test-name unet_course_demo
```

Expected outputs:

- `idownscale_output/runs/exp5/unet_course_demo/lightning_logs/version_best/metrics.csv`
- `idownscale_output/runs/exp5/unet_course_demo/lightning_logs/version_best/metrics_test_set.csv`
- checkpoint files under `checkpoints/`

Quick checks:

- look at loss evolution
- verify a best checkpoint exists
- inspect test-set metrics

## 8. Prediction phase

Command:

```bash
python bin/production/run_exp5_workflow.py \
  --exp exp5 \
  --steps predict_loop \
  --test-name unet_all \
  --simu-test gcm_bc
```

Expected outputs:

- prediction NetCDF under `idownscale_output/prediction/`

Quick checks:

- confirm time coverage
- confirm target grid shape
- inspect one time slice

## 9. Metrics phases

Command:

```bash
python bin/production/run_exp5_workflow.py \
  --exp exp5 \
  --steps metrics_day,metrics_month,value_metrics \
  --test-name unet_all \
  --simu-test gcm_bc
```

Expected outputs:

- daily CSV and NPZ
- monthly CSV and NPZ
- VALUE CSV

Quick checks:

- inspect the daily summary CSV
- inspect the monthly summary CSV
- inspect the VALUE summary table

## 10. Plotting phases

Command:

```bash
python bin/production/run_exp5_workflow.py \
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

## 11. Notebook use

The notebook should alternate:

1. explain the phase
2. show the command
3. list expected output files
4. embed a figure or table snippet
5. explain how to read the result

This makes the notebook useful both for reading and for staged reruns.
