# Dataset Files To Provide For The EGU26 Short Course

This page describes what should be published with the EGU26 short-course material.

The key practical decision is:

1. publish small project-specific artifacts directly from this repository
2. fetch large upstream climate datasets from their official repositories
3. generate local derived files such as cropped or reformatted products with repo tooling

This keeps the public package lighter and more durable while still supporting both
notebook-style demonstrations and training-oriented follow-up work.

For consistency with the validated workflow, the default scientific frame for the
course should remain aligned with `exp5`:

- same France domain
- same long historical and future climate windows
- same temperature case-study logic

This means some phases are inherently long-running. That should be explained openly in
the notebook rather than hidden behind unrealistic toy expectations.

See also:

- [Session materials](./SESSION_MATERIALS.md)
- [How to fetch upstream data](./HOW_TO_FETCH_UPSTREAM_DATA.md)

## 1. Recommended publication strategy

### Publish directly

These files are repo- or workflow-specific and are worth publishing with the short-course package:

- presentation PDF
- sample Jupyter notebook
- one checkpoint bundle:
  - `scratch/checkpoint_bundles/exp5_unet_all_bundle/`
  - optionally `scratch/checkpoint_bundles/exp5_swinunet_all_bundle/`
- selected metrics and plots for validation examples
- possibly a reduced training sample subset plus `statistics.json`

### Upstream raw archive: preferred source and optional mirror

These large files already live in public climate-data infrastructures and are better fetched from their official repositories:

- ERA5 reanalysis inputs
- CMIP6 raw GCM files
- E-OBS observational target data

The course material should explain how to retrieve them and how to turn them into the
local files expected by the repo.

For convenience, a Mercure mirror may also provide these raw files directly when we
want to simplify offline follow-up work.

### Generate locally

Some files used by this repo are not raw upstream products; they are project-side derivatives. Those should be generated locally from the fetched upstream data:

- France-cropped or standardized ERA5 working files
- France-cropped or otherwise prepared E-OBS target files
- bias-corrected GCM inputs
- Phase 1 training samples:
  - `sample_YYYYMMDD.npz`

## 2. Important distinction: upstream files vs local working files

### ERA5

The files:

- `rawdata/era5/tas_1d/tas_1d_<YEAR>_ERA5.nc`
- `rawdata/era5/orography_ERA5.nc`

can be treated as native ERA5-side inputs fetched from the official C3S/CDS distribution and placed into the repo layout.

Important note:

- `tas_1d_<YEAR>_ERA5.nc` is not the final France-cropped training-ready predictor by itself
- in the current workflow, ERA5 goes through standardization and spatial selection before entering the full reconstruction chain
- Phase 1 also uses a small bridge margin around the France domain before the conservative remapping chain

So for the course, we should document:

1. how to fetch the upstream ERA5 data from C3S/CDS
2. how to place those native files into the repo layout
3. how those files are then cropped/remapped for the demonstration or training workflow

### E-OBS

The target files:

- `rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc`
- `rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc`

represent the France-focused target side expected by `exp5`.

Important nuance:

- `rawdata/eobs/eobs_landseamask.nc` is not a France-focused file
- it is the Europe-scale E-OBS mask used by the workflow before cropping

For publication, we can either:

- provide these prepared France files directly, or
- document how to derive them from the upstream E-OBS distribution

### CMIP6

The raw GCM files:

- `rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc`
- `rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc`
- `rawdata/gcm/orog_Emon_CNRM-CM6-1_historical_r10i1p1f2_gr_185001-201412.nc`
- `rawdata/gcm/sftlf_fx_CNRM-CM6-1_historical_r1i1p1f2_gr.nc`

should ideally be fetched from official repositories when possible. A mirrored local
copy can still be made available through Mercure for convenience.

## 3. Minimal attendee package

This is the recommended public package for the short course.

### Files we should publish ourselves

- presentation PDF
- sample notebook
- one checkpoint bundle:
  - `scratch/checkpoint_bundles/exp5_unet_all_bundle/`
- selected validation outputs, for example:
  - pairwise distribution figure
  - monthly spatial bias figure
  - monthly spatial RMSE figure
  - VALUE-style summary

### Data we should explain how to fetch

- ERA5 temperature and orography from the official upstream services
- upstream E-OBS temperature, elevation, and mask
- upstream CMIP6 temperature, orography, and land fraction

### Local derivation step to explain

- how to create the repo-side France-focused files used by the notebook
- how to standardize longitudes and dimensions when needed
- how to crop or reformat the coarse ERA5 side before model use
- how to run the master workflow phase by phase rather than as one opaque block
- which diagnostic plots or summary tables should be checked after each phase

## 4. Training-capable package

If we want attendees or later users to be able to move beyond pretrained inference and toward retraining, then the material should also support the full Phase 1 data-preparation setup.

Training should be presented honestly:

- it needs substantial data
- it takes time
- it is not optional if the training world changes

### Recommended to publish

- `datasets/dataset_exp5_30y/statistics.json`
- optionally a reduced subset of:
  - `datasets/dataset_exp5_30y/sample_YYYYMMDD.npz`

### Reasonable alternatives

If distributing the full training-sample archive is too heavy, publish:

- `statistics.json`
- one checkpoint bundle
- one small sample subset for demonstration
- clear instructions showing how to rebuild the full sample archive from upstream data

That is enough for the short course while still making the retraining path understandable.

## 5. Optional derived products

These are useful but not mandatory for the public short-course bundle:

- corrected GCM files:
  - `rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_19800101-19991231_bc.nc`
  - `rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_20000101-20141231_bc.nc`
  - `rawdata/gcm/CNRM-CM6-1-BC/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231_bc.nc`
- prediction NetCDF examples
- selected files from `metrics/exp5/`
- selected files from `graph/metrics/exp5/`

These are most useful when we want attendees to inspect results without rerunning the full production chain.

## 6. France cropping and spatial preparation

The repo already contains reusable domain logic:

- `CONFIG['exp5']['domain'] = [-6.0, 10.0, 38.0, 54.0]`
- France grid registry in `data/spatial_registry.json`
- generic crop script:
  - `bin/preprocessing/crop_domain.py`

Important nuance:

- for the pedagogical notebook, a simple France crop may be enough
- for the full training path, the exact archival Phase 1 reconstruction is more subtle than a naive crop, because ERA5 is first handled on a small bridge domain before conservative remapping

So the course docs can already explain the general repo-side crop/reformat procedure,
while the exact training-grade France preparation can be documented more explicitly if
needed.

## 7. Practical publication advice

For the short course, the most realistic public release is:

- presentation PDF
- notebook
- one checkpoint bundle
- a few selected metrics/plots
- a reduced or minimal project-side demo subset if needed
- a clear data-acquisition guide pointing to official upstream sources

Optionally, the Mercure space can also expose a ready-made mirror of the larger raw
inputs for participants who prefer direct download over upstream retrieval.

That gives attendees a useful package without requiring us to redistribute a large climate archive.

## 8. Hosting split

The most practical deployment split is:

### GitHub repository

Use the main repository for:

- short-course documentation
- notebook
- presentation PDF
- instructions for fetching upstream ERA5 / CMIP6 / E-OBS data

### Shared file space for course artifacts

Use a separate file-sharing space only for the project-specific companion artifacts that
are useful to download directly:

- checkpoint bundle(s)
- possibly `statistics.json`
- possibly one or two prediction examples
- possibly selected diagnostic plots or metrics tables
- small France-specific prepared files, if we choose to provide them directly
- optionally, mirrored raw ERA5 / E-OBS / CMIP6 inputs for convenience

### Upstream repositories

The official upstream repositories remain the reference source for the large climate
archives:

- ERA5
- CMIP6
- E-OBS

The notebook and docs should explain how to fetch them and how to prepare the repo-side
working files, even if a Mercure mirror is also provided.

## 9. Rough storage estimates

The numbers below are approximate and are intended to help decide what should be placed
in a shared file space.

### A. If we include essentially everything local

This means:

- all raw ERA5 yearly files
- raw CMIP6 files
- prepared France E-OBS files
- corrected GCM files
- both checkpoint bundles
- the full `dataset_exp5_30y` archive
- metrics, plots, and example prediction NetCDFs

Rough total:

- about **42 GB**

This is too heavy for a normal course package and would mostly duplicate upstream data.

### B. Lean course package, without raw upstream ERA5 / CMIP6 / E-OBS

This means:

- notebook
- presentation PDF
- one checkpoint bundle
- `statistics.json`
- selected metrics / plots
- optionally one example prediction NetCDF
- optionally small France-specific prepared files

Rough total:

- about **0.2 to 0.5 GB**

Typical contributors:

- `exp5_unet_all_bundle`: about `90 MB`
- one example prediction NetCDF: about `172 MB`
- prepared France temperature target file: about `110 MB`
- metrics + plots: only a few MB

This is the best default package for a non-hands-on short course.

### C. Rich course companion package, still without raw upstream data

This means:

- both checkpoint bundles
- prepared France-specific small files
- corrected GCM files if desired
- example prediction NetCDFs
- metrics and plots
- optionally the full `dataset_exp5_30y` directory

Rough total:

- without the full `.npz` sample archive: about **0.6 to 0.8 GB**
- with the full `dataset_exp5_30y` archive: about **1.4 to 1.6 GB**

This is still manageable for a temporary shared space and gives people much more to
inspect offline.

## 10. Practical recommendation

For the short course as currently designed:

- **do not** ship raw ERA5 / CMIP6 / E-OBS
- **do not** ship the full output archive by default
- **do not** rely on the full `.npz` archive unless we later decide it is truly worth it

Instead:

- keep the notebook as the main reproducibility guide
- embed or show representative figures and tables inside the notebook
- provide a small set of project-specific artifacts separately
- let attendees rerun the longer steps offline if they want the full workflow

That is the most honest balance between scientific realism and practical distribution.
