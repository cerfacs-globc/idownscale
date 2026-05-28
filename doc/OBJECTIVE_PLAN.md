# Objective Plan

Last updated: 2026-05-28

## Current objective

Completed generalization scope for the exp5-style workflow so source selection is config-driven and can use:

- Reanalysis sources such as `era5` or `era6`
- Model sources such as `gcm_cnrm_cm6_1`, `rcm_aladin`, or `cordex`
- Observation/target sources such as `eobs`, `cerra`, `safran`, or `custom_obs`
- Existing family aliases `gcm` and `rcm` without breaking current runs

Current follow-up objective:

- preserve the above source generalization while validating that outputs remain
  both technically correct and scientifically credible on Grace
- run a full archival parity campaign for the ALADIN/`rcm` path against Zoé's
  historical `exp5` reference outputs under:
  - `/scratch/globc/garcia/idownscale`
  - `/archive2/globc/garcia/garcia_copy_scratch/idownscale`

## Current task status

- [x] Centralize source metadata in `iriscc/settings.py` via `SOURCE_CATALOG`
- [x] Make workflow output paths derive from source metadata instead of hardcoded filenames
- [x] Allow `build_dataset_pp.py` to resolve raw model files from direct source keys
- [x] Allow `build_dataset_bc.py` to accept direct source keys while preserving `gcm` and `rcm`
- [x] Keep `bias_correction_ibicus.py` aligned with configurable BC/GCM geometry references
- [x] Support `safran` and env-configured `custom_obs` as observation target sources
- [x] Generalize the main day/month/VALUE evaluation entrypoints to use source-aware sample/prediction resolution
- [x] Generalize remaining evaluation utilities that previously used hardcoded prediction globs
- [x] Add production workflow dispatch for `sbck_cdft` alongside `ibicus_cdft`
- [x] Validate end-to-end on Grace for at least one direct-source case and one alias case
- [x] Keep unsupported BC methods explicit; production methods are now `ibicus_cdft` and `sbck_cdft`
- [x] Make `bias_correction_ibicus.py` support a real lightweight `--test` mode for Grace smoke validation
- [x] Make `bias_correction_sbck.py` support a real lightweight `--test` mode for Grace smoke validation
- [x] Fix corrected-sample regeneration so smoke BC runs only iterate over dates present in the corrected dataset
- [x] Fix raw ALADIN source resolution to pick the file whose year span contains the requested date
- [x] Fix Grace raw RCM/ALADIN geometry handling for conservative regridding in `build_dataset_bc.py`
- [x] Fix Grace raw RCM/ALADIN geometry handling for conservative regridding in `build_dataset_pp.py`
- [ ] Run a full ALADIN archival parity test against Zoé's `exp5` reference, analogous to the earlier `exp5_full` parity work
- [ ] Extend validation from technical integrity to scientific acceptance for every parity run:
  - structural parity of BC bundles and packaged sample datasets
  - scientifically coherent diagnostics and VALUE-style historical validation
  - comparison against archival metrics/plots where available

## Branch checkpoint

- Active checkpoint branch for this recovery/validation cycle:
  - `grace-bc-parity-20260528`
- Reason:
  - preserve work across chat-history loss
  - make it safe to checkpoint commits and push incremental validation milestones

## Recovery context

If the chat history disappears after an SSH/browser reconnect, resume from this objective:

1. Confirm source resolution still accepts both aliases and direct model keys.
2. Confirm Grace smoke validation still passes for:
   - GCM alias raw/BC/corrected packaging
   - GCM direct-source raw/BC packaging
   - RCM alias raw/BC packaging
3. Use Zoé's archival ALADIN `exp5` outputs under `/scratch/globc/garcia/idownscale` as the main parity reference.
4. Run a full ALADIN parity test, not just file-existence checks:
   - BC bundle parity
   - raw packaged dataset parity
   - corrected packaged dataset parity
   - prediction and metrics parity where reference outputs exist
5. Resume on branch `grace-bc-parity-20260528` unless a newer checkpoint branch has been created.
5. Inspect the latest diffs in:
   - `iriscc/settings.py`
   - `iriscc/datautils.py`
   - `bin/preprocessing/build_dataset_bc.py`
   - `bin/preprocessing/build_dataset_pp.py`
   - `bin/preprocessing/bias_correction_ibicus.py`
   - `bin/preprocessing/bias_correction_sbck.py`
   - `bin/production/run_exp5_workflow.py`

## Grace validation plan

### Pre-flight

1. Activate the intended Grace environment and confirm `python`, `xarray`, `xesmf`, `torch`, and `ibicus` import cleanly.
2. Export any non-default roots needed for Grace, especially:
   - `IDOWNSCALE_RAW_DIR`
   - `IDOWNSCALE_OUTPUT_DIR`
   - `IDOWNSCALE_DATASET_BC_DIR`
   - Optional source-specific overrides such as `IDOWNSCALE_CORDEX_DIR`, `IDOWNSCALE_ERA6_DIR`, `IDOWNSCALE_CERRA_DIR`
   - For custom targets: `IDOWNSCALE_CUSTOM_OBS_DIR` and optionally `IDOWNSCALE_CUSTOM_OBS_GEOMETRY`, `IDOWNSCALE_CUSTOM_OBS_GLOB_PATTERN`, `IDOWNSCALE_CUSTOM_OBS_YEARLY_PATTERN`, `IDOWNSCALE_CUSTOM_OBS_MASK_TYPE`
3. Verify the target source files exist for the chosen experiment config.

### Target-source notes

- `target_source: 'safran'` already works through the observation loader path.
- `target_source: 'custom_obs'` now works when the corresponding `IDOWNSCALE_CUSTOM_OBS_*` environment variables describe the dataset layout.
- For `custom_obs`, keep `CONFIG[exp]['target_file']` and `CONFIG[exp]['orog_file']` aligned with the custom target grid used for reformatting and elevation input.

### Dry-run matrix

Run these first to confirm command construction and path expectations:

```bash
python bin/production/run_exp5_workflow.py --exp exp5 --simu gcm --dry-run
python bin/production/run_exp5_workflow.py --exp exp5 --simu rcm --dry-run
python bin/production/run_exp5_workflow.py --exp exp5 --simu gcm_cnrm_cm6_1 --dry-run
python bin/production/run_exp5_workflow.py --exp exp5 --simu cordex --dry-run
```

### Fast execution checks

Use a one-day or short-window smoke test before large jobs:

```bash
python bin/preprocessing/build_dataset_bc.py --exp exp5 --simu gcm_cnrm_cm6_1 --var tas --ssp ssp585 --test
python bin/preprocessing/build_dataset_bc.py --exp exp5 --simu cordex --var tas --ssp ssp585 --test
python bin/preprocessing/build_dataset_pp.py --exp exp5 --simu gcm_cnrm_cm6_1 --var tas --ssp ssp585 --test
python bin/preprocessing/build_dataset_pp.py --exp exp5 --simu cordex --var tas --ssp ssp585 --test
```

For corrected datasets after BC output exists:

```bash
python bin/preprocessing/build_dataset_pp.py --exp exp5 --simu gcm_cnrm_cm6_1 --var tas --ssp ssp585 --corrected --test
python bin/preprocessing/build_dataset_pp.py --exp exp5 --simu cordex --var tas --ssp ssp585 --corrected --test
```

### End-to-end candidate workflow

Alias path:

```bash
python bin/production/run_exp5_workflow.py --exp exp5 --simu gcm --steps bc_dataset,bc_apply,pp_dataset --if-exists overwrite
```

Direct-source path:

```bash
python bin/production/run_exp5_workflow.py --exp exp5 --simu cordex --steps bc_dataset,bc_apply,pp_dataset --if-exists overwrite
```

SBCK path:

```bash
python bin/production/run_exp5_workflow.py --exp exp5 --simu gcm --bc-method sbck_cdft --steps bc_dataset,bc_apply,pp_dataset --if-exists overwrite
python bin/production/run_exp5_workflow.py --exp exp5 --simu cordex --bc-method sbck_cdft --steps bc_dataset,bc_apply,pp_dataset --if-exists overwrite
```

### Validation checks

After each run, verify:

1. `bc_*.npz` files are created under `DATASET_BC_DIR` with the requested `simu` token.
2. Bias-corrected NetCDF outputs are created with source-derived names rather than hardcoded CNRM/ALADIN names.
3. `dataset_<exp>_test_<variant>/sample_*.npz` exists and opens cleanly.
4. Historical sample files contain both `x` and `y`; future files contain `x`.
5. Shapes and coordinates are consistent with the expected target grid.

## Current Grace status

- Working Grace environment recipe for mixed module/venv imports:
  - `IDOWNSCALE_EXTRA_PYTHONPATH=/softs/local_arm/Anaconda/2024.02-1/envs/gloenv_py3.11_arm/lib/python3.12/site-packages`
  - `ESMFMKFILE=/softs/local_arm/Anaconda/2024.02-1/envs/gloenv_py3.11_arm/lib/esmf.mk`
- Grace validation completed on 2026-05-28:
  - environment preflight `267292`: completed; `xarray`, `xesmf`, `torch`, `cartopy`, `ibicus`, and `SBCK` all imported successfully
  - GCM alias smoke BC `267301`: completed
  - GCM alias smoke raw PP `267302`: completed
  - GCM direct smoke BC `267305`: completed
  - GCM direct smoke raw PP `267306`: completed
  - direct `cordex` smoke BC `267307`: failed as expected with explicit missing-root guidance for `rawdata/rcm/CORDEX`
  - direct `cordex` smoke raw PP `267308`: failed as expected with explicit missing-source-file guidance for `source='cordex'`
  - GCM alias BC-only end-to-end smoke with `ibicus_cdft` `267375`: completed
  - GCM alias BC-only end-to-end smoke with `sbck_cdft` `267376`: completed
  - RCM alias smoke BC `267377`: completed after ALADIN geometry and file-selection fixes
  - RCM alias smoke raw PP `267393`: completed after ALADIN geometry standardization fixes
- Grace validation outputs collected under:
  - `/gpfs-calypso/scratch/globc/page/idownscale_output/validation_20260528`
- Technical output checks now confirmed on Grace:
  - GCM corrected smoke datasets produce `dataset_exp5_test_gcm_bc/sample_19800101.npz`, `sample_20000101.npz`, and `sample_20150101.npz`
  - RCM raw smoke datasets produce `dataset_exp5_test_rcm/sample_20000101.npz` and `sample_20150101.npz`
  - historical samples contain both `x` and `y`; future samples contain `x`
  - packaged sample shapes are `(2, 64, 64)` for inputs and `(1, 64, 64)` for targets
- Scientific/parity reference found for ALADIN:
  - `/scratch/globc/garcia/idownscale/datasets/dataset_bc/dataset_exp5_test_rcm`
  - `/scratch/globc/garcia/idownscale/datasets/dataset_bc/dataset_exp5_test_rcm_bc`
  - `/scratch/globc/garcia/idownscale/prediction/tas_day_ALADIN_*_exp5_*_rcm_bc.nc`
  - matching mirror also present under `/archive2/globc/garcia/garcia_copy_scratch/idownscale`
- Observed data-layout constraint:
  - the accessible rerun raw-data tree contains `rawdata/rcm/CORDEX-BC` but not a raw `rawdata/rcm/CORDEX` source directory
  - until a real CORDEX input root is available or `IDOWNSCALE_CORDEX_DIR` is pointed at one, direct-source CORDEX BC dataset generation cannot complete on Grace
- Loader/performance hardening completed:
  - RCM source resolution now raises explicit `FileNotFoundError` messages for missing roots, missing patterns, or unmatched year spans instead of raw `IndexError`
  - BC correction scripts no longer load a full GCM timeseries just to recover output coordinates
  - BC correction `--test` mode now skips expensive diagnostics and only exercises the core corrected-output path
- Reusable Grace helper:
  - `bin/production/run_in_grace_env.sh` mirrors the working mixed module/venv setup for standalone preprocessing and evaluation `sbatch` jobs

### Known limits

- Production BC methods are `ibicus_cdft` and `sbck_cdft`; other methods still raise explicitly.
- `sbck_cdft` additionally requires the `SBCK` Python package to be present in the Grace environment.
- Runtime validation still depends on the target Grace environment having `cartopy`, `xesmf`, `torch`, and the expected data roots available.
- Full ALADIN archival parity is not yet complete:
  - raw packaged RCM samples now match Zoé's reference structurally
  - BC bundle parity is not yet proven equivalent to Zoé's archival `rcm` reference
  - corrected ALADIN packaged dataset parity and prediction/metrics parity still need to be run

## Scientific validation method

For BC-only validation, do not stop at file existence or shape checks.
Use:

1. PDF/distribution comparison plots for raw vs BC vs reference.
2. Simple numerical summaries that are easy to audit:
   - mean
   - standard deviation
   - RMSE against ERA5 on historical periods
   - mean bias and q05/q50/q95 biases
3. Future-period sanity checks:
   - mean/std shifts relative to historical test
   - confirm BC does not create obviously implausible distribution drift

Implementation status:

- Added reusable validator:
  - `bin/evaluation/validate_bc_outputs.py`
- Outputs:
  - `metrics/<exp>/bc_validation_summary_<exp>_<simu>.csv`
  - `graph/metrics/<exp>/bc_validation_pdf_<exp>_<simu>.png`
  - optional archive parity CSVs for BC bundles and corrected packaged samples

Current results:

- GCM BC scientific validation on Grace completed in limited-overlap mode:
  - job `267483`
  - outputs:
    - `/gpfs-calypso/scratch/globc/page/idownscale_output/metrics/exp5/bc_validation_summary_exp5_gcm.csv`
    - `/gpfs-calypso/scratch/globc/page/idownscale_output/graph/metrics/exp5/bc_validation_pdf_exp5_gcm.png`
  - note:
    - current default GCM corrected NetCDFs only overlap one smoke day with the full BC bundles, so these numbers validate the method and not yet a full-period scientific GCM certification
- ALADIN/RCM BC dataset performance fix validated on Grace:
  - job `267491`
  - result:
    - optimized `build_dataset_bc.py` now processes by source-file batch instead of one day at a time
    - `--test` smoke for `simu=rcm` completed in about 10 seconds
  - status:
    - this is the active path for the new full ALADIN parity rerun

## Active Grace jobs

- Full ALADIN/RCM BC rerun in isolated parity root:
  - job `267492`
  - output root:
    - `/gpfs-calypso/scratch/globc/page/idownscale_output/validation_20260528/rcm_parity_full`
  - command intent:
    - `bc_dataset,bc_apply` for `simu=rcm`
- Dependent ALADIN/RCM scientific + archive parity validation:
  - job `267493`
  - starts after successful completion of job `267492`
