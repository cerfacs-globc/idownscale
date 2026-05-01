# Dataset Files To Provide For The EGU26 Short Course

This page lists the files published for the EGU26 short course and maps them to the current Mercure release layout.

The Mercure root is:

- `https://mercure.cerfacs.fr/egu26scml/`

The release is split into:

1. `required/` for the core attendee package
2. `nice_to_have/` for supplementary derived products
3. `raw_data/` for climate input files
4. `phase_outputs/` for workflow-generated Phase 1 products

## 1. Required attendee package

This is the recommended entrypoint for attendees.

The notebook is expected to support the full workflow:

1. pre-Phase-1 France target preparation and cropping
2. Phase 1 sample generation
3. statistics
4. bias correction
5. training or pretrained checkpoint reuse
6. prediction and evaluation

Attendees should be able to run the same logical steps on a laptop, workstation, or supercomputer. What changes by environment is mainly the amount of data, the selected date window, and the runtime cost of the heavier phases.

The notebook is not only a static course document. It should also give users an
easy executable entrypoint for the whole sequence:

1. prepare directories for raw inputs and generated outputs
2. run the France cropping and preparation step
3. run each later phase independently
4. inspect outputs, statistics, and plots after each phase

Some phases can take 5 to 6 hours in fuller runs, so the notebook should be
organized to remain useful both for reading and for staged re-execution.

Checkpoint reuse is optional. If attendees keep the same workflow setup, they can use the published checkpoint bundle. If they change the domain, configuration, variables, preprocessing, normalization, or related assumptions, they should plan to retrain.

Users also need:

- environment-setup guidance for Conda, `xesmf`, `ESMF`, `ESMFMKFILE`, `ibicus`, and `SBCK`
- helper scripts to prepare directory paths and runtime locations
- a reusable cropping command for the France preparation step
- phase-by-phase instructions showing which command to run and what to inspect
- phase-validation notes describing which tables, statistics, and diagnostic plots to check after each phase

See also:

- [Environment setup](./ENVIRONMENT_SETUP.md)
- [Helper scripts](./HELPER_SCRIPTS.md)
- [Workflow phases](./WORKFLOW_PHASES.md)
- [Phase validation](./PHASE_VALIDATION.md)

### Core notebook and inference artifacts

- `required/notebook/egu26_short_course_notebook.ipynb`
- `required/checkpoint_bundles/exp5_unet_all_bundle/`
- `required/predictions/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_all_gcm_bc.nc`
- `required/metrics/metrics_test_mean_daily_exp5_unet_all_gcm_bc.csv`
- `required/metrics/metrics_test_mean_monthly_exp5_unet_all_gcm_bc.csv`
- `required/metrics/value_metrics_exp5_unet_all.csv`
- `required/metrics/statistics.json`

These already-published statistics and metrics can be loaded directly in the
notebook for discussion, tables, and validation checks.

### Required plots

- `required/plots/daily_rmse_seasonal_unet_all_gcm_bc.png`
- `required/plots/monthly_rmse_seasonal_unet_all_gcm_bc.png`
- `required/plots/monthly_spatial_bias_distribution_unet_all_gcm_bc.png`
- `required/plots/monthly_spatial_rmse_distribution_unet_all_gcm_bc.png`
- `required/plots/exp5_pairwise_distribution_quantiles.png`
- `required/plots/exp5_historical_5curve_pdf.png`

These already-published figures can also be shown directly in the notebook when
the session uses precomputed outputs instead of regenerating every plot live.

### Required documentation mirror

- `required/docs/SESSION_SUMMARY.md`
- `required/docs/SESSION_MATERIALS.md`
- `required/docs/DATASETS_TO_PROVIDE.md`
- `required/docs/HOW_TO_FETCH_UPSTREAM_DATA.md`
- `required/docs/EXPECTED_PHASE_OUTPUTS.md`
- `required/docs/LOCAL_WORKFLOW_RUNBOOK.md`

## 2. Raw climate inputs

These files are published under `raw_data/`.

They are the starting point for attendees who want to execute the workflow themselves rather than only inspect precomputed outputs.

### ERA5

- `raw_data/era5/orography_ERA5.nc`
- `raw_data/era5/tas_1d/`

### E-OBS

- `raw_data/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc`
- `raw_data/eobs/elevation_ens_025deg_reg_v29_0e.nc`

These upstream E-OBS files are the basis for the pre-Phase-1 France cropping step that prepares the `exp5` target-side files.

### GCM

- `raw_data/gcm/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc`
- `raw_data/gcm/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc`
- `raw_data/gcm/orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc`
- `raw_data/gcm/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc`

## 3. Supplementary derived products

These files are published under `nice_to_have/`.

### Optional checkpoint bundle

- `nice_to_have/checkpoint_bundles/exp5_swinunet_all_bundle/`

### France-cropped target files

- `nice_to_have/eobs_france/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `nice_to_have/eobs_france/elevation_ens_025deg_reg_v29_0e_france.nc`
- `nice_to_have/eobs_france/eobs_landseamask.nc`

These files are useful as reference outputs for the pre-Phase-1 France preparation step and as shortcuts for attendees who want to compare their locally generated files against the published versions.

### Bias-corrected GCM files

- `nice_to_have/gcm_bc/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc`
- `nice_to_have/gcm_bc/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc`
- `nice_to_have/gcm_bc/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231_bc.nc`

### Additional diagnostics

- `nice_to_have/predictions/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_grace30_gcm_bc.nc`
- `nice_to_have/metrics/metrics_test_mean_daily_exp5_unet_grace30_gcm_bc.csv`
- `nice_to_have/metrics/metrics_test_mean_monthly_exp5_unet_grace30_gcm_bc.csv`
- `nice_to_have/metrics/value_metrics_exp5_unet_grace30.csv`
- selected files from `nice_to_have/plots/`

## 4. Phase outputs

The published Phase 1 workflow outputs are under:

- `phase_outputs/dataset_exp5_30y/`

This directory includes:

- `sample_YYYYMMDD.npz` daily Phase 1 training samples
- `hist_y_train.png`
- `hist_y_val.png`
- `hist_y_test.png`

These files are not only demonstration artifacts. They are also reference products for attendees who want to verify that their own Phase 1 workflow, after France cropping and preprocessing, is producing the expected outputs.

## 5. Release archives

Mercure also provides two packaged downloads:

- `egu26_sc_required.tar.gz`
- `egu26_sc_nice_to_have.tar.gz`

These tarballs mirror the `required/` and `nice_to_have/` trees for simpler bulk download.

They should be mentioned prominently in the notebook and setup notes because
they are the easiest way for users to retrieve the published course data before
starting the workflow.
