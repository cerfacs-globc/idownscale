# Expected Outputs By Workflow Phase

This page lists the most useful output files to show after each phase of the
`exp5` workflow in the EGU26 short-course notebook.

The intent is pedagogical:

- run one phase at a time with the master workflow
- inspect a small number of outputs immediately
- use those outputs to decide whether the phase worked as expected

## 1. Phase 1: build training samples

### Optional pre-Phase-1 target preparation

Master workflow step:

- `prep_phase1`

Expected outputs:

- `rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`

Optional convenience output if explicitly requested through the helper:

- `rawdata/eobs/eobs_landseamask_france.nc`

What to inspect:

- target grid shape should be `64x64`
- longitude centers should run from `-5.875` to `9.875`
- latitude centers should run from `38.125` to `53.875`
- file naming and location should match the `exp5` setup

## 2. Phase 1: build training samples

Master workflow step:

- `phase1`

Expected outputs:

- `datasets/dataset_exp5_30y/sample_YYYYMMDD.npz`

What to inspect:

- one or two sample files
- `x` and `y` array shapes
- channel count
- NaN / mask behavior

## 3. Statistics phase

Master workflow step:

- `stats`

Expected outputs:

- `datasets/dataset_exp5_30y/statistics.json`

What to inspect:

- variables / channels covered
- mean / std / min / max presence
- consistency with the intended predictor and target setup

## 4. Training phase

Master workflow step:

- `train`

Expected outputs:

- `runs/exp5/<test-name>/lightning_logs/version_best/hparams.yaml`
- `runs/exp5/<test-name>/lightning_logs/version_best/metrics.csv`
- `runs/exp5/<test-name>/lightning_logs/version_best/metrics_test_set.csv`
- `runs/exp5/<test-name>/lightning_logs/version_best/checkpoints/`

What to inspect:

- training history
- validation loss evolution
- presence of a best checkpoint
- test-set metrics after fit

Good visual companion:

- training loss history plot

## 5. Prediction phase

Master workflow step:

- `predict_loop`

Expected outputs:

- `prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_<test-name>_gcm_bc.nc`

Examples from validated runs:

- `prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_all_gcm_bc.nc`
- `prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_grace30_gcm_bc.nc`

What to inspect:

- time coverage
- grid shape
- variable name and units
- a quick map or summary statistic for one date

## 6. Daily metrics phase

Master workflow step:

- `metrics_day`

Expected outputs:

- `metrics/exp5/mean_metrics/metrics_test_mean_daily_exp5_<test-name>_gcm_bc.csv`
- `metrics/exp5/mean_metrics/metrics_test_daily_exp5_<test-name>_gcm_bc.npz`

Examples:

- `metrics_test_mean_daily_exp5_unet_all_gcm_bc.csv`
- `metrics_test_mean_daily_exp5_unet_grace30_gcm_bc.csv`

What to inspect:

- daily mean summary CSV
- seasonal RMSE evolution
- daily spatial bias / RMSE distributions

Good visual companions:

- `daily_rmse_seasonal_<test-name>_gcm_bc.png`
- `daily_spatial_bias_distribution_<test-name>_gcm_bc.png`
- `daily_spatial_rmse_distribution_<test-name>_gcm_bc.png`

## 7. Monthly metrics phase

Master workflow step:

- `metrics_month`

Expected outputs:

- `metrics/exp5/mean_metrics/metrics_test_mean_monthly_exp5_<test-name>_gcm_bc.csv`
- `metrics/exp5/mean_metrics/metrics_test_monthly_exp5_<test-name>_gcm_bc.npz`

Examples:

- `metrics_test_mean_monthly_exp5_unet_all_gcm_bc.csv`
- `metrics_test_mean_monthly_exp5_unet_grace30_gcm_bc.csv`

What to inspect:

- monthly summary CSV
- monthly RMSE seasonal figure
- monthly spatial bias / RMSE distributions

Good visual companions:

- `monthly_rmse_seasonal_<test-name>_gcm_bc.png`
- `monthly_spatial_bias_distribution_<test-name>_gcm_bc.png`
- `monthly_spatial_rmse_distribution_<test-name>_gcm_bc.png`

## 8. VALUE summary phase

Master workflow step:

- `value_metrics`

Expected outputs:

- `metrics/exp5/value_metrics_exp5_<test-name>.csv`

Examples:

- `value_metrics_exp5_unet_all.csv`
- `value_metrics_exp5_unet_grace30.csv`

What to inspect:

- marginal bias
- standard deviation ratio
- tail behavior
- spatial correlation
- spatial RMSE

## 9. Plotting phase

Master workflow steps:

- `plot_metrics_day`
- `plot_metrics_month`

Expected outputs:

- daily and monthly diagnostic figures under:
  - `graph/metrics/exp5/<test-name>_gcm_bc/`

Useful global comparison figures already available:

- `graph/metrics/exp5/exp5_pairwise_distribution_quantiles.png`
- `graph/metrics/exp5/exp5_historical_5curve_pdf.png`

## 10. Recommended notebook rhythm

For the short course, the notebook should ideally alternate:

1. explanation of the phase
2. master-script command
3. list of expected output files
4. one or two figures or table snippets
5. short interpretation of what success looks like

This is much more useful than showing the whole workflow as one giant command with no checks in between.
