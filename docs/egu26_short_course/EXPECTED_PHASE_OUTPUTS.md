# Expected Outputs By Workflow Phase

This page lists the most useful files and checks to show after each phase of the
`exp5` workflow.

## 1. France target preparation

Expected outputs:

- `rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`

What to inspect:

- target grid shape
- longitude and latitude bounds
- file naming and location

## 2. Phase 1 sample generation

Expected outputs:

- `idownscale_output/datasets/dataset_exp5_30y/sample_YYYYMMDD.npz`

What to inspect:

- one or two sample files
- `x` and `y` array shapes
- channel count
- NaN and mask behavior

## 3. Statistics phase

Expected outputs:

- `idownscale_output/datasets/dataset_exp5_30y/statistics.json`
- `idownscale_output/datasets/dataset_exp5_30y/hist_y_train.png`
- `idownscale_output/datasets/dataset_exp5_30y/hist_y_val.png`
- `idownscale_output/datasets/dataset_exp5_30y/hist_y_test.png`

What to inspect:

- variables and channels covered
- mean, std, min, and max entries
- histogram shape and range

## 4. Training phase

Expected outputs:

- `idownscale_output/runs/exp5/<test-name>/lightning_logs/version_best/hparams.yaml`
- `idownscale_output/runs/exp5/<test-name>/lightning_logs/version_best/metrics.csv`
- `idownscale_output/runs/exp5/<test-name>/lightning_logs/version_best/metrics_test_set.csv`
- `idownscale_output/runs/exp5/<test-name>/lightning_logs/version_best/checkpoints/`

What to inspect:

- training history
- validation loss evolution
- presence of a best checkpoint
- test-set metrics

## 5. Prediction phase

Expected outputs:

- `idownscale_output/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_<test-name>_gcm_bc.nc`

What to inspect:

- time coverage
- grid shape
- variable name and units
- one quick map or summary for one date

## 6. Daily metrics phase

Expected outputs:

- `idownscale_output/metrics/exp5/mean_metrics/metrics_test_mean_daily_exp5_<test-name>_gcm_bc.csv`
- `idownscale_output/metrics/exp5/mean_metrics/metrics_test_daily_exp5_<test-name>_gcm_bc.npz`

What to inspect:

- daily mean summary CSV
- seasonal RMSE evolution
- daily spatial bias and RMSE distributions

## 7. Monthly metrics phase

Expected outputs:

- `idownscale_output/metrics/exp5/mean_metrics/metrics_test_mean_monthly_exp5_<test-name>_gcm_bc.csv`
- `idownscale_output/metrics/exp5/mean_metrics/metrics_test_monthly_exp5_<test-name>_gcm_bc.npz`

What to inspect:

- monthly summary CSV
- monthly RMSE figure
- monthly spatial bias and RMSE distributions

## 8. VALUE summary phase

Expected outputs:

- `idownscale_output/metrics/exp5/value_metrics_exp5_<test-name>.csv`

What to inspect:

- marginal bias
- standard deviation ratio
- tail behavior
- spatial correlation
- spatial RMSE

## 9. Plotting phase

Expected outputs:

- daily and monthly diagnostic figures under
  `idownscale_output/graph/metrics/exp5/`

Useful companion figures already available from published outputs:

- `exp5_pairwise_distribution_quantiles.png`
- `exp5_historical_5curve_pdf.png`

## 10. Recommended notebook rhythm

For each phase:

1. explain the phase
2. show the command
3. list expected outputs
4. show one or two figures or tables
5. explain what success looks like
