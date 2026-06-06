# Perfect-Model Implementation Notes

Date: 2026-06-05

This note keeps the implementation and recovery history separate from the
science-facing perfect-model report.

## 1. Main bugs discovered

### Wrong BC reference in perfect-model mode

- degraded ALADIN was being bias-corrected against `ERA5`
- BC and ML outputs were then being evaluated against native ALADIN
- this created a false scientific diagnosis that the BC baseline was broken

Fix:

- `CONFIG['perfect_model_rcm']['bc_reanalysis_source'] = 'rcm_aladin'`

### Corrected perfect-model sample writer rebuilt `y` through the wrong path

- corrected sample packaging still used the generic observation target loader
- this caused:
  - `IndexError: 2-dimensional boolean indexing is not supported`

Fix:

- corrected sample writers in:
  - `bin/preprocessing/bias_correction_ibicus.py`
  - `bin/preprocessing/bias_correction_sbck.py`
- now rebuild perfect-model pseudo-truth through the same native-model path as
  `build_dataset_pp.py`

### Corrected sample packaging OOM

- `1887337` failed with `OUT_OF_MEMORY` after corrected BC NetCDFs had already
  been written

Technical cause:

- repeated orography loading
- temporary regridded day datasets not closed promptly

Fix:

- load orography once
- package corrected samples through a shared helper
- close each one-day regridded dataset immediately
- close corrected datasets after packaging

### Corrected eval dataset contamination

- `build_dataset_pp.py` included `train_hist` whenever `--corrected` was set

Effect:

- eval dataset wrongly included `1980-1999`

Fix:

- corrected mode no longer implies `include_train_hist`
- only explicit `--include-train-hist` adds the train historical block

### Future sample-dir resolution bug

- `predict_loop.py` resolved future perfect-model runs to the wrong sample
  directory

Fix:

- date-aware sample-dir resolution was added to `bin/training/predict_loop.py`

### Same hidden assumption in future diagnostics

- future sample directory was also assumed implicitly in:
  - `bin/evaluation/plot_perfect_model_distribution_pdf.py`
  - `bin/evaluation/compare_perfect_model_climate_signal.py`

Fix:

- both scripts now resolve sample directories per requested window

## 2. Recovery-wave lessons

### Why the first future relaunches failed

The corrected future helper dataset on disk was not a full all-window future
diagnostic dataset.

Observed sample coverage in:

- `dataset_perfect_model_rcm_validation_windows_rcm_bc`

was exactly:

- `19800101-20141231`
- `20900101-21001231`

So the first recovery waves failed because they assumed that these windows were
also present:

- `20150101-20291231`
- `20300101-20441231`
- `20450101-20591231`
- `20600101-20741231`
- `20750101-20891231`

They were not.

### Important distinction

This does **not** mean that the future corrected NetCDF does not exist.

What it means is:

- the special validation helper sample tree used for perfect-model diagnostics
  was only materialized for historical plus late-century
- therefore only the late-century future window could be used in the bounded
  recovery

## 3. Final successful recovery jobs

- corrected BC rebuild / dataset rebuild:
  - `1888048`
- late-century future recovery:
  - `1888548` `pm_bc8_uon`
  - `1888549` `pm_bc8_unet`
  - `1888550` `pm_bc8_rep3`
  - `1888551` `pm_bc8_mini`
  - `1888552` `pm_bc8_seed2`
- dependent post-processing:
  - `1888553`

## 4. Files changed during the recovery

Main files touched for the recovery itself:

- `iriscc/settings.py`
- `bin/preprocessing/build_dataset_pp.py`
- `bin/preprocessing/bias_correction_ibicus.py`
- `bin/preprocessing/bias_correction_sbck.py`
- `bin/training/predict_loop.py`
- `bin/evaluation/plot_perfect_model_distribution_pdf.py`
- `bin/evaluation/compare_perfect_model_climate_signal.py`

## 5. Operational note

A stale repo-local fallback output tree appeared under:

- `/scratch/globc/page/idownscale_rerun/idownscale_output`

Cause:

- some runs were executed without exporting
  `IDOWNSCALE_OUTPUT_DIR=/scratch/globc/page/idownscale_output`

Status:

- this stale local tree was removed on 2026-06-05

## 6. Remaining implementation need

If we want complete future-period perfect-model diagnostics, we should define a
canonical corrected future sample workflow that materializes all intended
future windows with target-bearing sample packaging, not only the historical
blocks plus the late-century helper window.
