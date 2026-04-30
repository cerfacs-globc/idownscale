# Mercure Deployment Plan For The EGU26 Short Course

This note describes a practical folder structure and copy commands for deploying the
short-course material to Mercure.

Current target:

- remote path: `/data/projects/egu26scml`
- public HTTPS URL: `https://mercure.cerfacs.fr/egu26scml/`

Recommended local roots when running the copy commands:

- repo root: `/scratch/globc/page/idownscale_rerun`
- output root: `/scratch/globc/page/idownscale_output`

If you are not already in the repo root, either `cd /scratch/globc/page/idownscale_rerun`
first, or replace relative paths like `docs/...`, `rawdata/...`, and `scratch/...` with
their absolute form under `/scratch/globc/page/idownscale_rerun/`.

The plan is intentionally split into four levels:

1. `required`
2. `nice_to_have`
3. `raw_data`
4. `phase_outputs`

## 1. Recommended remote structure

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

## 2. Create the Mercure folders

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

## 3. Required tier

This is the smallest tier that makes the notebook and the story useful.

### Source files

Docs:

- `/scratch/globc/page/idownscale_rerun/docs/egu26_short_course/SESSION_MATERIALS.md`
- `/scratch/globc/page/idownscale_rerun/docs/egu26_short_course/SESSION_SUMMARY.md`
- `/scratch/globc/page/idownscale_rerun/docs/egu26_short_course/DATASETS_TO_PROVIDE.md`
- `/scratch/globc/page/idownscale_rerun/docs/egu26_short_course/HOW_TO_FETCH_UPSTREAM_DATA.md`
- `/scratch/globc/page/idownscale_rerun/docs/egu26_short_course/EXPECTED_PHASE_OUTPUTS.md`
- `/scratch/globc/page/idownscale_rerun/docs/egu26_short_course/LOCAL_WORKFLOW_RUNBOOK.md`

Notebook:

- `/scratch/globc/page/idownscale_rerun/docs/egu26_short_course/egu26_short_course_notebook.ipynb`

Checkpoint bundle:

- `/scratch/globc/page/idownscale_rerun/scratch/checkpoint_bundles/exp5_unet_all_bundle/`

Statistics:

- `/scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/statistics.json`

Core metrics:

- `/scratch/globc/page/idownscale_output/metrics/exp5/mean_metrics/metrics_test_mean_daily_exp5_unet_all_gcm_bc.csv`
- `/scratch/globc/page/idownscale_output/metrics/exp5/mean_metrics/metrics_test_mean_monthly_exp5_unet_all_gcm_bc.csv`
- `/scratch/globc/page/idownscale_output/metrics/exp5/value_metrics_exp5_unet_all.csv`

Core plots:

- `/scratch/globc/page/idownscale_output/graph/metrics/exp5/exp5_pairwise_distribution_quantiles.png`
- `/scratch/globc/page/idownscale_output/graph/metrics/exp5/exp5_historical_5curve_pdf.png`
- `/scratch/globc/page/idownscale_output/graph/metrics/exp5/unet_all_gcm_bc/daily_rmse_seasonal_unet_all_gcm_bc.png`
- `/scratch/globc/page/idownscale_output/graph/metrics/exp5/unet_all_gcm_bc/monthly_rmse_seasonal_unet_all_gcm_bc.png`
- `/scratch/globc/page/idownscale_output/graph/metrics/exp5/unet_all_gcm_bc/monthly_spatial_bias_distribution_unet_all_gcm_bc.png`
- `/scratch/globc/page/idownscale_output/graph/metrics/exp5/unet_all_gcm_bc/monthly_spatial_rmse_distribution_unet_all_gcm_bc.png`

Optional but useful example prediction:

- `/scratch/globc/page/idownscale_output/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_all_gcm_bc.nc`

Presentation:

- add the final PDF here when ready

### Copy commands

```bash
scp /scratch/globc/page/idownscale_rerun/docs/egu26_short_course/SESSION_MATERIALS.md mercure:/data/projects/egu26scml/required/docs/
scp /scratch/globc/page/idownscale_rerun/docs/egu26_short_course/SESSION_SUMMARY.md mercure:/data/projects/egu26scml/required/docs/
scp /scratch/globc/page/idownscale_rerun/docs/egu26_short_course/DATASETS_TO_PROVIDE.md mercure:/data/projects/egu26scml/required/docs/
scp /scratch/globc/page/idownscale_rerun/docs/egu26_short_course/HOW_TO_FETCH_UPSTREAM_DATA.md mercure:/data/projects/egu26scml/required/docs/
scp /scratch/globc/page/idownscale_rerun/docs/egu26_short_course/EXPECTED_PHASE_OUTPUTS.md mercure:/data/projects/egu26scml/required/docs/
scp /scratch/globc/page/idownscale_rerun/docs/egu26_short_course/LOCAL_WORKFLOW_RUNBOOK.md mercure:/data/projects/egu26scml/required/docs/
scp /scratch/globc/page/idownscale_rerun/docs/egu26_short_course/egu26_short_course_notebook.ipynb mercure:/data/projects/egu26scml/required/notebook/

scp /scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/statistics.json mercure:/data/projects/egu26scml/required/metrics/

scp /scratch/globc/page/idownscale_output/metrics/exp5/mean_metrics/metrics_test_mean_daily_exp5_unet_all_gcm_bc.csv mercure:/data/projects/egu26scml/required/metrics/
scp /scratch/globc/page/idownscale_output/metrics/exp5/mean_metrics/metrics_test_mean_monthly_exp5_unet_all_gcm_bc.csv mercure:/data/projects/egu26scml/required/metrics/
scp /scratch/globc/page/idownscale_output/metrics/exp5/value_metrics_exp5_unet_all.csv mercure:/data/projects/egu26scml/required/metrics/

scp /scratch/globc/page/idownscale_output/graph/metrics/exp5/exp5_pairwise_distribution_quantiles.png mercure:/data/projects/egu26scml/required/plots/
scp /scratch/globc/page/idownscale_output/graph/metrics/exp5/exp5_historical_5curve_pdf.png mercure:/data/projects/egu26scml/required/plots/
scp /scratch/globc/page/idownscale_output/graph/metrics/exp5/unet_all_gcm_bc/daily_rmse_seasonal_unet_all_gcm_bc.png mercure:/data/projects/egu26scml/required/plots/
scp /scratch/globc/page/idownscale_output/graph/metrics/exp5/unet_all_gcm_bc/monthly_rmse_seasonal_unet_all_gcm_bc.png mercure:/data/projects/egu26scml/required/plots/
scp /scratch/globc/page/idownscale_output/graph/metrics/exp5/unet_all_gcm_bc/monthly_spatial_bias_distribution_unet_all_gcm_bc.png mercure:/data/projects/egu26scml/required/plots/
scp /scratch/globc/page/idownscale_output/graph/metrics/exp5/unet_all_gcm_bc/monthly_spatial_rmse_distribution_unet_all_gcm_bc.png mercure:/data/projects/egu26scml/required/plots/

scp /scratch/globc/page/idownscale_output/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_all_gcm_bc.nc mercure:/data/projects/egu26scml/required/predictions/

scp -r /scratch/globc/page/idownscale_rerun/scratch/checkpoint_bundles/exp5_unet_all_bundle mercure:/data/projects/egu26scml/required/checkpoint_bundles/
```

## 4. Nice-to-have tier

This tier makes offline follow-up easier without mirroring the whole climate archive.

### Source files

Prepared France-side files:

- `/scratch/globc/page/idownscale_rerun/rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `/scratch/globc/page/idownscale_rerun/rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`

Optional convenience-only file, not required by the main workflow:

- `/scratch/globc/page/idownscale_rerun/rawdata/eobs/eobs_landseamask_france.nc`

Bias-corrected GCM files:

- `/scratch/globc/page/idownscale_rerun/rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc`
- `/scratch/globc/page/idownscale_rerun/rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc`
- `/scratch/globc/page/idownscale_rerun/rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231_bc.nc`

Second checkpoint bundle:

- `/scratch/globc/page/idownscale_rerun/scratch/checkpoint_bundles/exp5_swinunet_all_bundle/`

Second example prediction:

- `/scratch/globc/page/idownscale_output/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_grace30_gcm_bc.nc`

Additional metrics:

- `/scratch/globc/page/idownscale_output/metrics/exp5/mean_metrics/metrics_test_mean_daily_exp5_unet_grace30_gcm_bc.csv`
- `/scratch/globc/page/idownscale_output/metrics/exp5/mean_metrics/metrics_test_mean_monthly_exp5_unet_grace30_gcm_bc.csv`
- `/scratch/globc/page/idownscale_output/metrics/exp5/value_metrics_exp5_unet_grace30.csv`

Additional plots:

- all files from:
  - `/scratch/globc/page/idownscale_output/graph/metrics/exp5/unet_grace30_gcm_bc/`

### Copy commands

```bash
scp /scratch/globc/page/idownscale_rerun/rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc mercure:/data/projects/egu26scml/nice_to_have/eobs_france/
scp /scratch/globc/page/idownscale_rerun/rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc mercure:/data/projects/egu26scml/nice_to_have/eobs_france/
scp /scratch/globc/page/idownscale_rerun/rawdata/eobs/eobs_landseamask_france.nc mercure:/data/projects/egu26scml/nice_to_have/eobs_france/

scp /scratch/globc/page/idownscale_rerun/rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc mercure:/data/projects/egu26scml/nice_to_have/gcm_bc/
scp /scratch/globc/page/idownscale_rerun/rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc mercure:/data/projects/egu26scml/nice_to_have/gcm_bc/
scp /scratch/globc/page/idownscale_rerun/rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231_bc.nc mercure:/data/projects/egu26scml/nice_to_have/gcm_bc/

scp -r /scratch/globc/page/idownscale_rerun/scratch/checkpoint_bundles/exp5_swinunet_all_bundle mercure:/data/projects/egu26scml/nice_to_have/checkpoint_bundles/

scp /scratch/globc/page/idownscale_output/prediction/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101_20141231_exp5_unet_grace30_gcm_bc.nc mercure:/data/projects/egu26scml/nice_to_have/predictions/

scp /scratch/globc/page/idownscale_output/metrics/exp5/mean_metrics/metrics_test_mean_daily_exp5_unet_grace30_gcm_bc.csv mercure:/data/projects/egu26scml/nice_to_have/metrics/
scp /scratch/globc/page/idownscale_output/metrics/exp5/mean_metrics/metrics_test_mean_monthly_exp5_unet_grace30_gcm_bc.csv mercure:/data/projects/egu26scml/nice_to_have/metrics/
scp /scratch/globc/page/idownscale_output/metrics/exp5/value_metrics_exp5_unet_grace30.csv mercure:/data/projects/egu26scml/nice_to_have/metrics/

scp /scratch/globc/page/idownscale_output/graph/metrics/exp5/unet_grace30_gcm_bc/* mercure:/data/projects/egu26scml/nice_to_have/plots/
```

## 5. Raw-data tier

This is only needed if you want Mercure to act as a local mirror of the climate inputs.

### Source directories and files

ERA5:

- `/scratch/globc/page/idownscale_rerun/rawdata/era5/tas_1d/`
- `/scratch/globc/page/idownscale_rerun/rawdata/era5/orography_ERA5.nc`

E-OBS:

- `/scratch/globc/page/idownscale_rerun/rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc`
- `/scratch/globc/page/idownscale_rerun/rawdata/eobs/elevation_ens_025deg_reg_v29_0e.nc`

GCM:

- `/scratch/globc/page/idownscale_rerun/rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc`
- `/scratch/globc/page/idownscale_rerun/rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc`
- `/scratch/globc/page/idownscale_rerun/rawdata/gcm/orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc`
- `/scratch/globc/page/idownscale_rerun/rawdata/gcm/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc`

### Copy commands

For the large raw tier, `rsync` is preferable to `scp`.

```bash
rsync -av /scratch/globc/page/idownscale_rerun/rawdata/era5/tas_1d/ mercure:/data/projects/egu26scml/raw_data/era5/tas_1d/
scp /scratch/globc/page/idownscale_rerun/rawdata/era5/orography_ERA5.nc mercure:/data/projects/egu26scml/raw_data/era5/

scp /scratch/globc/page/idownscale_rerun/rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc mercure:/data/projects/egu26scml/raw_data/eobs/
scp /scratch/globc/page/idownscale_rerun/rawdata/eobs/elevation_ens_025deg_reg_v29_0e.nc mercure:/data/projects/egu26scml/raw_data/eobs/

scp /scratch/globc/page/idownscale_rerun/rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc mercure:/data/projects/egu26scml/raw_data/gcm/
scp /scratch/globc/page/idownscale_rerun/rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc mercure:/data/projects/egu26scml/raw_data/gcm/
scp /scratch/globc/page/idownscale_rerun/rawdata/gcm/orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc mercure:/data/projects/egu26scml/raw_data/gcm/
scp /scratch/globc/page/idownscale_rerun/rawdata/gcm/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc mercure:/data/projects/egu26scml/raw_data/gcm/
```

## 6. Phase-output tier

This is for deeper offline reproducibility.

### Source directory

- `/scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/`

### Copy commands

If you want the full archive:

```bash
rsync -av /scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/ mercure:/data/projects/egu26scml/phase_outputs/dataset_exp5_30y/
```

If you want only `statistics.json` and a few samples:

```bash
scp /scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/statistics.json mercure:/data/projects/egu26scml/phase_outputs/dataset_exp5_30y/
scp /scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/sample_19800101.npz mercure:/data/projects/egu26scml/phase_outputs/dataset_exp5_30y/
scp /scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/sample_20000101.npz mercure:/data/projects/egu26scml/phase_outputs/dataset_exp5_30y/
```

## 7. Optional archive creation

If later you decide to provide compact downloadable bundles for the small tiers, create
the folders first on Mercure, then optionally add tarballs locally and upload them.

Examples:

- `egu26_sc_required.tar.gz`
- `egu26_sc_nice_to_have.tar.gz`

This is optional now that the Mercure URL is directly browsable.
