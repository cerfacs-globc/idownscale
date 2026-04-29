# Dataset Files To Provide For The EGU26 Short Course

This page lists the files that are most useful to publish alongside the short-course material.

The list is split into:

1. a minimal attendee package for the demonstration notebook
2. a fuller reproducibility package for users who want to rerun more of the workflow

The exact download links can be added later.

## 1. Minimal attendee package

This is the recommended public package for the short course.

### Core climate inputs

- `rawdata/era5/tas_1d/tas_1d_<YEAR>_ERA5.nc`
  - Daily ERA5 near-surface temperature files for the years actually used in the notebook
- `rawdata/era5/orography_ERA5.nc`
  - ERA5 orography used on the coarse predictor side
- `rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
  - High-resolution E-OBS temperature target over France
- `rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`
  - E-OBS elevation over France
- `rawdata/eobs/eobs_landseamask.nc`
  - E-OBS land-sea mask

### CMIP6 sample inputs

- `rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc`
  - Raw historical GCM temperature
- `rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc`
  - Raw future GCM temperature
- `rawdata/gcm/orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc`
  - GCM orography
- `rawdata/gcm/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc`
  - GCM land fraction mask

### If the notebook demonstrates corrected inputs directly

- `rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc`
- `rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc`
- `rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231_bc.nc`

### If the notebook demonstrates pretrained inference

Provide at least one checkpoint bundle:

- `scratch/checkpoint_bundles/exp5_unet_all_bundle/`
- optionally `scratch/checkpoint_bundles/exp5_swinunet_all_bundle/`

This is preferable to distributing a bare checkpoint file because the bundle also carries the manifest and required configuration context.

## 2. Full reproducibility package

This larger package is for users who want to rerun the workflow more fully rather than only follow the demonstration notebook.

### Training-data contract

- Phase 1 sample files or a reduced subset of them:
  - `datasets/dataset_exp5_30y/sample_YYYYMMDD.npz`
- associated normalization file:
  - `datasets/dataset_exp5_30y/statistics.json`

If distributing the full Phase 1 sample archive is too heavy, provide:

- a reduced sample subset for demonstration
- `statistics.json`
- a checkpoint bundle

### Outputs for validation examples

If you want attendees to inspect precomputed results without rerunning everything:

- prediction NetCDF example(s) from `prediction/`
- selected metrics files from `metrics/exp5/`
- selected plots from `doc/egu_preview_assets/` or `graph/metrics/exp5/`

Useful examples include:

- pairwise distribution figure
- monthly spatial bias figure
- monthly spatial RMSE figure
- VALUE-style summary

## 3. Practical publication advice

For the short course, the most realistic public release is usually:

- presentation PDF
- notebook
- minimal climate input subset
- one checkpoint bundle
- a few selected precomputed outputs for validation

That is enough to make the material useful without requiring attendees to mirror the full project storage layout.

## 4. Files to decide case by case

These files may or may not need to be published depending on how ambitious the notebook is:

- raw bias-correction intermediate products
- full 30-year Phase 1 training sample archive
- all prediction NetCDF outputs
- all daily/monthly metrics arrays

If the short course is primarily pedagogical, these are usually optional.
