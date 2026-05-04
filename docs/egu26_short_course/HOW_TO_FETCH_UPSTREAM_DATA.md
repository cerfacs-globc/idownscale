# How To Fetch Upstream Data For The EGU26 Short Course

This note explains how to retrieve the large upstream climate datasets used by
the short-course workflow without depending on local project storage.

Use this page together with:

- [DATA_SETUP_QUICKSTART.md](./DATA_SETUP_QUICKSTART.md)
- [SESSION_MATERIALS.md](./SESSION_MATERIALS.md)

## 1. General rule

- fetch raw climate data from the official upstream repositories
- generate local working files with `idownscale` tooling
- use Mercure mainly for lighter project-specific artifacts

## 2. ERA5

Official entry points:

- `https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview`
- `https://confluence.ecmwf.int/display/CKB/How%2Bto%2Bdownload%2BERA5?src=contextnavpagetreemode`

Variables of interest for the temperature case study:

- `2m_temperature`
- `geopotential` or equivalent field used for orography context

Typical local working files:

- `rawdata/era5/tas_1d/tas_1d_<YEAR>_ERA5.nc`
- `rawdata/era5/orography_ERA5.nc`

## 3. E-OBS

Official entry points:

- `https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php`
- `https://dataplatform.knmi.nl/dataset/e-obs-all-variables-1-0`

Typical upstream inputs:

- `rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc`
- `rawdata/eobs/elevation_ens_025deg_reg_v29_0e.nc`

France-cropped files used later in the workflow:

- `rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`

To prepare them locally, use:

```bash
python bin/preprocessing/crop_domain.py \
  --input rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc \
  --output rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc \
  --exp exp5 \
  --standardize
```

```bash
python bin/preprocessing/crop_domain.py \
  --input rawdata/eobs/elevation_ens_025deg_reg_v29_0e.nc \
  --output rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc \
  --exp exp5 \
  --standardize
```

## 4. CMIP6

Primary route:

- `https://www.copernicus.eu/en/access-data/copernicus-services-catalogue/cmip6-climate-projections`

Fallback route:

- `https://esgf-ui.ceda.ac.uk/`
- `https://esgf.github.io/esgf-user-support/`

Typical files for the temperature case study:

- daily `tas` for historical
- daily `tas` for `ssp585`
- fixed fields such as `orog` and `sftlf`

Typical local working files:

- `rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc`
- `rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc`
- `rawdata/gcm/orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc`
- `rawdata/gcm/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc`

## 5. After download

Once the upstream files are available locally:

1. create the repo-side directory tree with
   `bash bin/production/setup_egu26_short_course_tree.sh .`
2. place upstream files under `rawdata/`
3. prepare the France-cropped E-OBS targets if needed
4. continue with the phase-by-phase workflow

Use these pages next:

- [LOCAL_WORKFLOW_RUNBOOK.md](./LOCAL_WORKFLOW_RUNBOOK.md)
- [EXPECTED_PHASE_OUTPUTS.md](./EXPECTED_PHASE_OUTPUTS.md)
