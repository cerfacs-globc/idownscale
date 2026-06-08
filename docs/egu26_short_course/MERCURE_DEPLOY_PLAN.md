# Mercure Deployment Plan For The EGU26 Short Course

This note describes a practical folder structure and copy commands for deploying the
short-course material to Mercure.

Current target:

- remote path: `/data/projects/egu26scml`
- public HTTPS URL: `https://mercure.cerfacs.fr/egu26scml/`

Recommended local roots when running the copy commands:

```bash
export REPO_ROOT=/path/to/idownscale
export RUNTIME_ROOT=/path/to/idownscale_runtime
export RAWDATA_ROOT=$RUNTIME_ROOT/rawdata
export OUTPUT_ROOT=$RUNTIME_ROOT/output
export GRAPHS_ROOT=$RUNTIME_ROOT/graphs
export WORK_MATERIALS_ROOT=/path/to/idownscale_work_materials
```

The checkpoint bundles used for the course may live outside the runtime tree. If
they were preserved from a previous working checkout, point `WORK_MATERIALS_ROOT`
to that preserved material before copying.

The plan is intentionally split into four levels:

1. `required`
2. `nice_to_have`
3. `raw_data`
4. `phase_outputs`

## 1. Recommended Remote Structure

```text
/data/projects/egu26scml/
├── required/
│   ├── docs/
│   ├── notebook/
│   ├── checkpoint_bundles/
│   ├── metrics/
│   ├── plots/
│   ├── predictions/
│   └── presentation/
├── nice_to_have/
│   ├── eobs_france/
│   ├── gcm_bc/
│   ├── checkpoint_bundles/
│   ├── predictions/
│   ├── metrics/
│   └── plots/
├── raw_data/
│   ├── era5/
│   ├── eobs/
│   └── gcm/
└── phase_outputs/
    └── dataset_exp5_30y/
```

## 2. Create The Mercure Folders

```bash
ssh mercure '
mkdir -p /data/projects/egu26scml/required/docs
mkdir -p /data/projects/egu26scml/required/notebook
mkdir -p /data/projects/egu26scml/required/checkpoint_bundles
mkdir -p /data/projects/egu26scml/required/metrics
mkdir -p /data/projects/egu26scml/required/plots
mkdir -p /data/projects/egu26scml/required/predictions
mkdir -p /data/projects/egu26scml/required/presentation
mkdir -p /data/projects/egu26scml/nice_to_have/eobs_france
mkdir -p /data/projects/egu26scml/nice_to_have/gcm_bc
mkdir -p /data/projects/egu26scml/nice_to_have/checkpoint_bundles
mkdir -p /data/projects/egu26scml/nice_to_have/predictions
mkdir -p /data/projects/egu26scml/nice_to_have/metrics
mkdir -p /data/projects/egu26scml/nice_to_have/plots
mkdir -p /data/projects/egu26scml/raw_data/era5
mkdir -p /data/projects/egu26scml/raw_data/eobs
mkdir -p /data/projects/egu26scml/raw_data/gcm
mkdir -p /data/projects/egu26scml/phase_outputs/dataset_exp5_30y
'
```

## 3. Required Tier

This is the smallest tier that makes the notebook and the story useful.

### Source Files

Docs:

- `$REPO_ROOT/docs/egu26_short_course/SESSION_MATERIALS.md`
- `$REPO_ROOT/docs/egu26_short_course/SESSION_SUMMARY.md`
- `$REPO_ROOT/docs/egu26_short_course/DATASETS_TO_PROVIDE.md`
- `$REPO_ROOT/docs/egu26_short_course/HOW_TO_FETCH_UPSTREAM_DATA.md`
- `$REPO_ROOT/docs/egu26_short_course/EXPECTED_PHASE_OUTPUTS.md`
- `$REPO_ROOT/docs/egu26_short_course/LOCAL_WORKFLOW_RUNBOOK.md`

Notebook:

- `$REPO_ROOT/docs/egu26_short_course/egu26_short_course_notebook.ipynb`

Checkpoint bundle:

- `$WORK_MATERIALS_ROOT/checkpoint_bundles/exp5_unet_all_bundle/`

Statistics:

- `$OUTPUT_ROOT/datasets/dataset_exp5_30y/statistics.json`

Core metrics:

- `$OUTPUT_ROOT/metrics/exp5/mean_metrics/metrics_test_mean_daily_exp5_unet_all_gcm_bc.csv`
- `$OUTPUT_ROOT/metrics/exp5/mean_metrics/metrics_test_mean_monthly_exp5_unet_all_gcm_bc.csv`
- `$OUTPUT_ROOT/metrics/exp5/value_metrics_exp5_unet_all.csv`

Core plots:

- `$GRAPHS_ROOT/metrics/exp5/exp5_pairwise_distribution_quantiles.png`
- `$GRAPHS_ROOT/metrics/exp5/exp5_historical_5curve_pdf.png`
- `$GRAPHS_ROOT/metrics/exp5/unet_all_gcm_bc/daily_rmse_seasonal_unet_all_gcm_bc.png`
- `$GRAPHS_ROOT/metrics/exp5/unet_all_gcm_bc/monthly_rmse_seasonal_unet_all_gcm_bc.png`
- `$GRAPHS_ROOT/metrics/exp5/unet_all_gcm_bc/monthly_spatial_bias_distribution_unet_all_gcm_bc.png`
- `$GRAPHS_ROOT/metrics/exp5/unet_all_gcm_bc/monthly_spatial_rmse_distribution_unet_all_gcm_bc.png`

Optional but useful example prediction:

- `$OUTPUT_ROOT/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_all_gcm_bc.nc`

Presentation:

- add the final PDF here when ready

### Copy Commands

```bash
scp "$REPO_ROOT"/docs/egu26_short_course/SESSION_MATERIALS.md mercure:/data/projects/egu26scml/required/docs/
scp "$REPO_ROOT"/docs/egu26_short_course/SESSION_SUMMARY.md mercure:/data/projects/egu26scml/required/docs/
scp "$REPO_ROOT"/docs/egu26_short_course/DATASETS_TO_PROVIDE.md mercure:/data/projects/egu26scml/required/docs/
scp "$REPO_ROOT"/docs/egu26_short_course/HOW_TO_FETCH_UPSTREAM_DATA.md mercure:/data/projects/egu26scml/required/docs/
scp "$REPO_ROOT"/docs/egu26_short_course/EXPECTED_PHASE_OUTPUTS.md mercure:/data/projects/egu26scml/required/docs/
scp "$REPO_ROOT"/docs/egu26_short_course/LOCAL_WORKFLOW_RUNBOOK.md mercure:/data/projects/egu26scml/required/docs/
scp "$REPO_ROOT"/docs/egu26_short_course/egu26_short_course_notebook.ipynb mercure:/data/projects/egu26scml/required/notebook/

scp "$OUTPUT_ROOT"/datasets/dataset_exp5_30y/statistics.json mercure:/data/projects/egu26scml/required/metrics/

scp "$OUTPUT_ROOT"/metrics/exp5/mean_metrics/metrics_test_mean_daily_exp5_unet_all_gcm_bc.csv mercure:/data/projects/egu26scml/required/metrics/
scp "$OUTPUT_ROOT"/metrics/exp5/mean_metrics/metrics_test_mean_monthly_exp5_unet_all_gcm_bc.csv mercure:/data/projects/egu26scml/required/metrics/
scp "$OUTPUT_ROOT"/metrics/exp5/value_metrics_exp5_unet_all.csv mercure:/data/projects/egu26scml/required/metrics/

scp "$GRAPHS_ROOT"/metrics/exp5/exp5_pairwise_distribution_quantiles.png mercure:/data/projects/egu26scml/required/plots/
scp "$GRAPHS_ROOT"/metrics/exp5/exp5_historical_5curve_pdf.png mercure:/data/projects/egu26scml/required/plots/
scp "$GRAPHS_ROOT"/metrics/exp5/unet_all_gcm_bc/daily_rmse_seasonal_unet_all_gcm_bc.png mercure:/data/projects/egu26scml/required/plots/
scp "$GRAPHS_ROOT"/metrics/exp5/unet_all_gcm_bc/monthly_rmse_seasonal_unet_all_gcm_bc.png mercure:/data/projects/egu26scml/required/plots/
scp "$GRAPHS_ROOT"/metrics/exp5/unet_all_gcm_bc/monthly_spatial_bias_distribution_unet_all_gcm_bc.png mercure:/data/projects/egu26scml/required/plots/
scp "$GRAPHS_ROOT"/metrics/exp5/unet_all_gcm_bc/monthly_spatial_rmse_distribution_unet_all_gcm_bc.png mercure:/data/projects/egu26scml/required/plots/

scp "$OUTPUT_ROOT"/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_all_gcm_bc.nc mercure:/data/projects/egu26scml/required/predictions/

scp -r "$WORK_MATERIALS_ROOT"/checkpoint_bundles/exp5_unet_all_bundle mercure:/data/projects/egu26scml/required/checkpoint_bundles/
```

## 4. Nice-To-Have Tier

This tier makes offline follow-up easier without mirroring the whole climate archive.

### Source Files

Prepared France-side files:

- `$RAWDATA_ROOT/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `$RAWDATA_ROOT/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`

Optional convenience-only file, not required by the main workflow:

- `$RAWDATA_ROOT/eobs/eobs_landseamask_france.nc`

Bias-corrected GCM files:

- `$RAWDATA_ROOT/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc`
- `$RAWDATA_ROOT/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc`
- `$RAWDATA_ROOT/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231_bc.nc`

Second checkpoint bundle:

- `$WORK_MATERIALS_ROOT/checkpoint_bundles/exp5_swinunet_all_bundle/`

Second example prediction:

- `$OUTPUT_ROOT/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_grace30_gcm_bc.nc`

Additional metrics:

- `$OUTPUT_ROOT/metrics/exp5/mean_metrics/metrics_test_mean_daily_exp5_unet_grace30_gcm_bc.csv`
- `$OUTPUT_ROOT/metrics/exp5/mean_metrics/metrics_test_mean_monthly_exp5_unet_grace30_gcm_bc.csv`
- `$OUTPUT_ROOT/metrics/exp5/value_metrics_exp5_unet_grace30.csv`

Additional plots:

- all files from `$GRAPHS_ROOT/metrics/exp5/unet_grace30_gcm_bc/`

### Copy Commands

```bash
scp "$RAWDATA_ROOT"/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc mercure:/data/projects/egu26scml/nice_to_have/eobs_france/
scp "$RAWDATA_ROOT"/eobs/elevation_ens_025deg_reg_v29_0e_france.nc mercure:/data/projects/egu26scml/nice_to_have/eobs_france/
scp "$RAWDATA_ROOT"/eobs/eobs_landseamask_france.nc mercure:/data/projects/egu26scml/nice_to_have/eobs_france/

scp "$RAWDATA_ROOT"/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc mercure:/data/projects/egu26scml/nice_to_have/gcm_bc/
scp "$RAWDATA_ROOT"/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc mercure:/data/projects/egu26scml/nice_to_have/gcm_bc/
scp "$RAWDATA_ROOT"/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231_bc.nc mercure:/data/projects/egu26scml/nice_to_have/gcm_bc/

scp -r "$WORK_MATERIALS_ROOT"/checkpoint_bundles/exp5_swinunet_all_bundle mercure:/data/projects/egu26scml/nice_to_have/checkpoint_bundles/

scp "$OUTPUT_ROOT"/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_grace30_gcm_bc.nc mercure:/data/projects/egu26scml/nice_to_have/predictions/

scp "$OUTPUT_ROOT"/metrics/exp5/mean_metrics/metrics_test_mean_daily_exp5_unet_grace30_gcm_bc.csv mercure:/data/projects/egu26scml/nice_to_have/metrics/
scp "$OUTPUT_ROOT"/metrics/exp5/mean_metrics/metrics_test_mean_monthly_exp5_unet_grace30_gcm_bc.csv mercure:/data/projects/egu26scml/nice_to_have/metrics/
scp "$OUTPUT_ROOT"/metrics/exp5/value_metrics_exp5_unet_grace30.csv mercure:/data/projects/egu26scml/nice_to_have/metrics/

scp "$GRAPHS_ROOT"/metrics/exp5/unet_grace30_gcm_bc/* mercure:/data/projects/egu26scml/nice_to_have/plots/
```

## 5. Raw-Data Tier

This is only needed if you want Mercure to act as a local mirror of the climate inputs.

### Source Directories And Files

ERA5:

- `$RAWDATA_ROOT/era5/tas_1d/`
- `$RAWDATA_ROOT/era5/orography_ERA5.nc`

E-OBS:

- `$RAWDATA_ROOT/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc`
- `$RAWDATA_ROOT/eobs/elevation_ens_025deg_reg_v29_0e.nc`

GCM:

- `$RAWDATA_ROOT/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc`
- `$RAWDATA_ROOT/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc`
- `$RAWDATA_ROOT/gcm/orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc`
- `$RAWDATA_ROOT/gcm/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc`

### Copy Commands

For the large raw tier, `rsync` is preferable to `scp`.

```bash
rsync -av "$RAWDATA_ROOT"/era5/tas_1d/ mercure:/data/projects/egu26scml/raw_data/era5/tas_1d/
scp "$RAWDATA_ROOT"/era5/orography_ERA5.nc mercure:/data/projects/egu26scml/raw_data/era5/

scp "$RAWDATA_ROOT"/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc mercure:/data/projects/egu26scml/raw_data/eobs/
scp "$RAWDATA_ROOT"/eobs/elevation_ens_025deg_reg_v29_0e.nc mercure:/data/projects/egu26scml/raw_data/eobs/

scp "$RAWDATA_ROOT"/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc mercure:/data/projects/egu26scml/raw_data/gcm/
scp "$RAWDATA_ROOT"/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc mercure:/data/projects/egu26scml/raw_data/gcm/
scp "$RAWDATA_ROOT"/gcm/orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc mercure:/data/projects/egu26scml/raw_data/gcm/
scp "$RAWDATA_ROOT"/gcm/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc mercure:/data/projects/egu26scml/raw_data/gcm/
```

## 6. Phase-Output Tier

This is for deeper offline reproducibility.

### Source Directory

- `$OUTPUT_ROOT/datasets/dataset_exp5_30y/`

### Copy Commands

If you want the full archive:

```bash
rsync -av "$OUTPUT_ROOT"/datasets/dataset_exp5_30y/ mercure:/data/projects/egu26scml/phase_outputs/dataset_exp5_30y/
```

If you want only `statistics.json` and a few samples:

```bash
scp "$OUTPUT_ROOT"/datasets/dataset_exp5_30y/statistics.json mercure:/data/projects/egu26scml/phase_outputs/dataset_exp5_30y/
scp "$OUTPUT_ROOT"/datasets/dataset_exp5_30y/sample_19800101.npz mercure:/data/projects/egu26scml/phase_outputs/dataset_exp5_30y/
scp "$OUTPUT_ROOT"/datasets/dataset_exp5_30y/sample_20000101.npz mercure:/data/projects/egu26scml/phase_outputs/dataset_exp5_30y/
```

## 7. Optional Archive Creation

If later you decide to provide compact downloadable bundles for the small tiers, create
the folders first on Mercure, then optionally add tarballs locally and upload them.

Examples:

- `egu26_sc_required.tar.gz`
- `egu26_sc_nice_to_have.tar.gz`

This is optional now that the Mercure URL is directly browsable.
