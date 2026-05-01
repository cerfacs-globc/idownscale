# Phase Validation For The EGU26 Short Course

This page lists simple checks attendees can use to verify that each workflow
phase completed successfully.

The short-course notebook should use the same rhythm for every phase:

1. explain the scientific purpose of the phase
2. show the command to run
3. inspect one or two output files
4. show a small table or plot
5. discuss what a successful result looks like

This structure is important because some phases are long-running. Users should
be able to re-execute one phase, inspect the outputs, and stop there before
continuing later.

## 1. Pre-Phase-1 France preparation

Check:

- the output files exist
- the cropped files match the expected France domain
- longitude and latitude coordinates are ordered correctly

Useful files to inspect:

- `rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`

Useful notebook outputs:

- one quick map of the cropped temperature field
- one quick map of the cropped elevation field
- a short printed summary of dimensions and coordinate bounds

## 2. Phase 1 sample generation

Check:

- daily `sample_YYYYMMDD.npz` files are written
- `x` and `y` arrays have the expected shapes
- values look finite where expected

Useful files to inspect:

- one or two `sample_YYYYMMDD.npz` files

Useful notebook outputs:

- a small table showing array names, shapes, and basic min/max values
- one or two example maps from the predictor and target arrays

## 3. Statistics phase

Check:

- `statistics.json` exists
- the expected variables are present
- mean, standard deviation, min, and max values are populated

Useful files to inspect:

- `statistics.json`
- `hist_y_train.png`
- `hist_y_val.png`
- `hist_y_test.png`

Useful notebook outputs:

- a short printed table extracted from `statistics.json`
- the histogram figures

If the statistics phase is not rerun live, the notebook can reuse the published
`statistics.json`, `hist_y_train.png`, `hist_y_val.png`, and `hist_y_test.png`
files from Mercure.

## 4. Bias-correction dataset phase

Check:

- the `.npz` files for the bias-correction stage exist
- historical and future coarse inputs are both present
- array dimensions are consistent with the intended coarse-grid workflow

Useful files to inspect:

- `bc_train_hist_gcm.npz`
- `bc_test_hist_gcm.npz`
- `bc_test_future_gcm.npz`

Useful notebook outputs:

- a short table of array shapes
- one quick visualization of a coarse historical field

## 5. Bias-correction application phase

Check:

- corrected GCM NetCDF files are written
- the expected historical and future periods are present
- corrected fields look physically reasonable

Useful files to inspect:

- corrected historical and future GCM files

Useful notebook outputs:

- one side-by-side comparison of raw and corrected coarse inputs
- a short table of date coverage and variable names

## 6. Training phase

Check:

- the run directory is created
- training logs and metrics files exist
- a best checkpoint is saved

Useful files to inspect:

- `hparams.yaml`
- `metrics.csv`
- `metrics_test_set.csv`
- the checkpoint directory

Useful notebook outputs:

- a training-loss curve
- a short metrics table for train, validation, and test summaries

## 7. Prediction phase

Check:

- the prediction NetCDF file exists
- the date range is correct
- the grid and variable names are as expected

Useful files to inspect:

- one historical prediction file

Useful notebook outputs:

- one map for a selected date
- a short table with dimensions, time coverage, and variable names

## 8. Daily and monthly metrics phases

Check:

- metrics CSV and NPZ files exist
- daily and monthly summaries are both produced
- seasonal and spatial plots are generated

Useful files to inspect:

- `metrics_test_mean_daily_*.csv`
- `metrics_test_mean_monthly_*.csv`
- daily and monthly plot files

Useful notebook outputs:

- a compact summary table from the CSV files
- daily RMSE seasonal plot
- monthly RMSE seasonal plot
- spatial bias and RMSE distributions
- selected prediction-result figures shown alongside the diagnostics

The notebook can reuse the published metrics CSV files and plot files directly
when a live rerun would be too long for the session.

## 9. VALUE metrics phase

Check:

- the VALUE metrics CSV exists
- the main indicators are populated

Useful files to inspect:

- `value_metrics_*.csv`

Useful notebook outputs:

- a short table of VALUE metrics
- a short discussion of bias, spread, correlation, and RMSE behavior

At the end of the notebook, these evaluation outputs should be shown together:

- representative prediction-result maps
- bias plots
- daily and monthly diagnostic plots
- VALUE summary tables or figures

These can come either from a live run or from the existing published outputs on
Mercure.
