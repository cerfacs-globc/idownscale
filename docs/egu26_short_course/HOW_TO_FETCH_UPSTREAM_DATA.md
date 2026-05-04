# How To Fetch Upstream Data For The EGU26 Short Course

This note explains how to retrieve the large upstream climate datasets used by the
short-course workflow without redistributing them from local project storage.

Some of these upstream files may also be mirrored in the Mercure course space for
convenience. Even in that case, the official upstream repositories remain the
reference source, and this note documents the reproducible retrieval path.

The guiding principle is:

- fetch raw climate data from the official upstream repositories
- generate local repo-side working files with `idownscale` tooling
- publish only the lighter project-specific artifacts ourselves

For the concrete local directory layout and the order in which users should unpack
Mercure tar files and place data, see:

- [Data setup quickstart](./DATA_SETUP_QUICKSTART.md)

For the short course, the intended default scientific frame should stay aligned with
the validated `exp5` setup:

- same France domain
- same temperature case study
- same long historical / future climate windows

That means the workflow is not artificially tiny. Some phases take time because climate
training and climate-change inference genuinely require long periods.

## 1. Recommended source hierarchy

### Primary route

Use the official Copernicus / C3S / CDS services whenever they provide the required data:

- ERA5 from the Climate Data Store
- CMIP6 climate projections from the Copernicus climate-data catalogue when available
- E-OBS from the official Copernicus / KNMI distribution

This is the most stable teaching route for the short course.

### Secondary route

Use ESGF as a fallback for files that are not straightforward to retrieve from the
Copernicus path.

This is especially relevant during transitions in the ESGF ecosystem, where older
search APIs and client tools may still work but are clearly evolving.

## 2. ERA5

### Upstream source

Official sources:

- ERA5 daily statistics on single levels
- ERA5 hourly data on single levels

Useful official entry points:

- `https://www.copernicus.eu/en/access-data/copernicus-services-catalogue/era5-post-processed-daily-statistics-single-levels-1940`
- `https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview`
- `https://confluence.ecmwf.int/display/CKB/How%2Bto%2Bdownload%2BERA5?src=contextnavpagetreemode`

### Variables of interest

For the temperature case study:

- `2m_temperature`
- `geopotential` or equivalent field used to derive orography/elevation context

### Local repo-side working files

The repo expects local files such as:

- `rawdata/era5/tas_1d/tas_1d_<YEAR>_ERA5.nc`
- `rawdata/era5/orography_ERA5.nc`

Important note:

- these files should be treated as local working files in the repo layout
- `tas_1d_<YEAR>_ERA5.nc` is not by itself the final training-ready France predictor

### Local derivation step

The repo already contains a generic crop script:

- `bin/preprocessing/crop_domain.py`

and a dedicated helper for the `exp5` France target files:

- `bin/preprocessing/prepare_exp5_france_targets.py`

and the current `exp5` France domain is:

- `[-6.0, 10.0, 38.0, 54.0]`

For a simple notebook-grade France crop, the route is approximately:

```bash
python bin/preprocessing/crop_domain.py \
  --input rawdata/era5/tas_1d/tas_1d_1980_ERA5.nc \
  --output rawdata/era5/tas_1d_france/tas_1d_1980_ERA5_france.nc \
  --exp exp5 \
  --standardize
```

The same generic crop script can also be used with an explicit bounding box:

```bash
python bin/preprocessing/crop_domain.py \
  --input rawdata/era5/tas_1d/tas_1d_1980_ERA5.nc \
  --output rawdata/era5/tas_1d_france/tas_1d_1980_ERA5_france.nc \
  --domain -6 10 38 54 \
  --standardize
```

Important nuance for the full training path:

- the current Phase 1 reconstruction logic is more subtle than a naive France crop
- `build_dataset.py` currently uses a small bridge margin around the `exp5` France
  box before the conservative remapping chain

So for publication we can already document the generic crop/reformat route, while the
exact training-grade recipe can be documented more explicitly later if needed.

## 3. E-OBS

### Upstream source

Official sources:

- `https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php`
- `https://dataplatform.knmi.nl/dataset/e-obs-all-variables-1-0`
- `https://www.copernicus.eu/en/access-data/copernicus-services-catalogue/e-obs-daily-gridded-meteorological-data-europe-1950`

### Local repo-side target files

The repo uses France-focused target files such as:

- `rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`
- `rawdata/eobs/eobs_landseamask.nc`

Important nuance:

- `rawdata/eobs/eobs_landseamask.nc` remains a Europe-scale workflow input
- it is not itself a France-focused target file

For the short course, we can either:

- provide these prepared files directly, or
- document how to derive them from the upstream E-OBS distribution

The current `exp5` logic is:

- `Data.get_eobs_dataset()` starts from the raw E-OBS Europe-scale file
- it applies the E-OBS land-sea mask while still on the Europe-scale grid
- it then crops to the manually chosen France domain so that the target becomes `64x64`
- `build_dataset.py` then adds the elevation map, itself cropped on the France domain
- `TARGET_EOBS_FRANCE_FILE` is used as the target grid for regridding ERA5 onto the E-OBS geometry

Comparison of the full Europe file and the France file shows that the France target is
consistent with an exact `64x64` coordinate subset of the original Europe-scale
E-OBS file.

The inferred France target coordinates are:

- longitude centers from `-5.875` to `9.875`
- latitude centers from `38.125` to `53.875`

which corresponds to the manually chosen `exp5` France box:

- `[-6.0, 10.0, 38.0, 54.0]`

So the intended workflow can now be described more concretely:

1. fetch the original Europe-scale E-OBS data
2. apply the mask on the Europe-scale grid
3. crop to the France domain used by `exp5`
4. obtain a `64x64` France target grid with:
   - lon centers `-5.875 ... 9.875`
   - lat centers `38.125 ... 53.875`
5. use the resulting France target file as the target geometry for ERA5 remapping

### Commands for France target preparation

The simplest reproducible route is to use the dedicated helper:

```bash
python bin/preprocessing/prepare_exp5_france_targets.py
```

This prepares:

- `rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`

If the Europe-scale source files are already present locally, the same step is also
available through the workflow runner:

```bash
python bin/production/run_exp5_workflow.py \
  --exp exp5 \
  --steps prep_phase1
```

By default, existing France-focused files are kept. To overwrite them explicitly:

```bash
python bin/preprocessing/prepare_exp5_france_targets.py --force
```

An optional France-focused mask subset can also be generated for convenience:

```bash
python bin/preprocessing/prepare_exp5_france_targets.py \
  --include-mask
```

This convenience mask is not required by the main `exp5` workflow.

## 4. CMIP6

### Upstream source

Primary route:

- `https://www.copernicus.eu/en/access-data/copernicus-services-catalogue/cmip6-climate-projections`

Fallback route:

- `https://esgf-ui.ceda.ac.uk/`
- `https://esgf.github.io/esgf-user-support/`

### Variables and files of interest

For the current temperature case study:

- daily `tas` for historical
- daily `tas` for `ssp585`
- fixed fields such as `orog` and `sftlf`

Local working files in the repo currently include:

- `rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc`
- `rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc`
- `rawdata/gcm/orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc`
- `rawdata/gcm/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc`

## 5. What we should publish ourselves

Instead of mirroring the large upstream climate archive, we should publish:

- presentation PDF
- sample notebook
- one checkpoint bundle
- selected metrics/plots
- possibly a reduced training-sample subset
- instructions explaining how to fetch and derive the large climate files

## 6. What the notebook should explain

The notebook or its companion documentation should explain:

1. where to fetch ERA5, E-OBS, and CMIP6 upstream
2. which variables and files are needed
3. how to place them into the repo layout
4. how to standardize and crop/reformat them locally
5. how to move from raw inputs to:
   - notebook-ready inference examples
   - and, optionally, training-ready sample generation

## 7. Future refinement

Before final publication, we should still document more explicitly:

- the exact command history or script originally used to create `TARGET_EOBS_FRANCE_FILE`
- the exact recipe used for the France-ready elevation and mask companions
- whether we want the notebook to show those derivation steps explicitly or provide the resulting France files directly

But we can already build almost all of the course material around the strategy described in this document.
