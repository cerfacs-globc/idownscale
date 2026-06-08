# Objective Plan

Last updated: 2026-06-03

Kraken default runtime for this branch:

- interpreter: `/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1/bin/python`
- activation: `source /scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1/bin/activate`
- prefer this env for interactive verification and Slurm wrappers unless a
  task explicitly documents a different environment

## Current objective

Completed generalization scope for the exp5-style workflow so source selection is config-driven and can use:

- Reanalysis sources such as `era5` or `era6`
- Model sources such as `gcm_cnrm_cm6_1`, `rcm_aladin`, or `cordex`
- Observation/target sources such as `eobs`, `cerra`, `safran`, or `custom_obs`
- Existing family aliases `gcm` and `rcm` without breaking current runs

Current follow-up objective:

- preserve the above source generalization while validating that outputs remain
  both technically correct and scientifically credible on the active HPC target
- run a full archival parity campaign for the ALADIN/`rcm` path against Zoé's
  historical `exp5` reference outputs under:
  - `/scratch/globc/garcia/idownscale`
  - `/archive2/globc/garcia/garcia_copy_scratch/idownscale`
- turn the ALADIN perfect-model workflow into a standalone reproducible run so
  ML-vs-RCM comparisons can be executed without relying on the broader archival
  parity harness
- use Kraken as the primary execution target now that Grace is heavily
  congested

Current merge strategy:

- keep ALADIN/RCM stabilization work separate from the broader source
  generalization work
- only upstream the generalization framework after ALADIN/RCM support is judged
  stable enough against Zoé's reference with tolerance
- keep standalone perfect-model support in its own future branch/PR
- keep the Kraken migration and path normalization practical and incremental,
  rather than mixing it into the scientific PRs

## Validation protocol

Use this order before spending time on downstream expensive jobs such as GPU
prediction or full metric campaigns:

1. **Implementation/runtime verification**
   - command uses the intended environment, branch, roots, dates, scenario, and
     source aliases
   - Slurm jobs complete cleanly with explicit memory/time requests
   - no hidden defaults point to Calypso/Grace paths or stale historical-period
     assumptions
   - parallel runs do not share fragile mutable artifacts such as xESMF weight
     files unless they are known read-safe
2. **Structural data integrity**
   - expected date inventory is complete and has no extra dates
   - variables, shapes, dtypes, masks, calendars, and coordinates match the
     processing contract
   - historical perfect-model samples contain both `x` and `y`; future samples
     contain `x` unless a dedicated future-target mode is being validated
   - boundary dates around historical/future splits are present and consistent
3. **Scientific data-value validation**
   - temperature ranges, NaN masks, seasonal behavior, and CDF/quantile summaries
     are compatible with the expected physical/data-processing story
   - predictor fields must not silently repeat across historical/future periods;
     compare same calendar-day samples across periods to detect stale,
     tiled, or climatology-like input files
   - historical windows are checked against available targets or references
     using bias, RMSE, quantiles, correlations, and distribution plots
   - future windows are checked for temporal continuity, plausible warming or
     scenario behavior, and absence of artificial jumps introduced by file
     boundaries, remapping, or batching
4. **Comparison and acceptance**
   - compare against Zoé's archive where the archive is internally consistent
   - compare ML vs raw RCM for perfect-model experiments
   - document tolerances, known archive inconsistencies, and residual risks
     before treating a run as scientifically usable

This follows the usual V&V/UQ framing for numerical simulations: verification
checks that we implemented and ran the calculation correctly; validation checks
that the outputs are credible for the scientific use case; uncertainty and
sensitivity notes define what differences we accept and why.

Workflow implication:

- validation must be gated phase by phase, not only run at the end
- each expensive phase should have a cheap immediate verification gate before
  the next expensive phase starts
- a final report is still useful, but it should not be the first time we learn
  that phase 1, BC preparation, sample packaging, or prediction wrote invalid
  data
- the EGU course/runbook already follows this principle informally; production
  workflow support should make it explicit

Near-term workflow adjustment to implement:

- add lightweight `verify_*` steps or an optional `--verify-after-step` mode to
  `bin/production/run_exp5_workflow.py`
- gate `prep_phase1` with target/orography domain, coordinate, and finite-value
  checks
- gate `phase1` with date inventory, `x/y` shape, mask, and basic value-range
  checks
- gate `stats` with finite min/max/mean/std entries and expected train/val/test
  windows
- gate `bc_dataset` with NPZ key, shape, date-window, NaN-mask, and CDF/quantile
  checks for raw vs reference historical data
- gate `bc_apply` with corrected NetCDF coverage plus raw/reference/corrected
  CDF and quantile summaries before generating corrected samples or prediction
- gate `raw_dataset` and `pp_dataset` with sample inventory, boundary-date, and
  value-distribution checks
- gate `predict_loop` with NetCDF coverage, coordinate/shape checks, finite
  masks, and plausible tas ranges before metrics are launched
- gate metrics/report phases by checking that key metrics are finite and within
  documented tolerance/acceptance ranges

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
- [x] Run a full ALADIN archival parity test against Zoé's `exp5` reference, analogous to the earlier `exp5_full` parity work
- [x] Extend validation from technical integrity to scientific acceptance for every parity run:
  - structural parity of BC bundles and packaged sample datasets
  - scientifically coherent diagnostics and VALUE-style historical validation
  - comparison against archival metrics/plots where available
- [x] Extract a standalone ALADIN perfect-model run path for direct ML-vs-RCM comparison
- [x] Build and validate a core Kraken x86 GPU environment on `gpua30`
- [x] Confirm Kraken GPU usability with `torch` on `NVIDIA A30`
- [x] Fix the standalone perfect-model raw-RCM metrics path for cropped `64x64` samples
- [x] Normalize Kraken output layout so it mirrors the Calypso structured tree (`runs/`, `metrics/`, `datasets/`, `prediction/`)
- [x] Build future perfect-model raw RCM samples on Kraken
- [x] Run future perfect-model ML predictions against raw RCM inputs on Kraken
- [x] Build future ML-vs-RCM comparison tables once predictions exist

## Split map for future PRs

### ALADIN/RCM stabilization

Focus:
- correctness of ALADIN/RCM preprocessing, BC, corrected outputs, and
  scientific validation against Zoé with tolerance

Current branch files mostly in this bucket:
- `iriscc/datautils.py`
- `bin/preprocessing/build_dataset_bc.py`
- `bin/preprocessing/bias_correction_ibicus.py`
- `bin/preprocessing/bias_correction_sbck.py`
- `bin/evaluation/validate_bc_outputs.py`
- `bin/production/probe_rcm_future_parity_modes.py`

### Generalization framework

Focus:
- config-driven support for multiple reanalyses, observation targets, and model
  sources

Current branch files mostly in this bucket:
- `iriscc/settings.py`
- `bin/production/run_exp5_workflow.py`
- `bin/production/run_exp5_workflow_grace.sh`
- `bin/production/submit_exp5_workflow_grace.sh`
- `bin/production/run_in_grace_env.sh`

### Perfect-model standalone support

Focus:
- standalone ALADIN perfect-model run for direct ML-vs-RCM comparison

Current branch files likely feeding this bucket:
- parts of `bin/preprocessing/build_dataset_pp.py`
- `bin/production/run_exp5_perfect_model.py`
- `bin/production/submit_exp5_perfect_model_grace.sh`

### Mixed files to split later

- `iriscc/settings.py`
- `bin/production/run_exp5_workflow.py`
- `bin/preprocessing/build_dataset_pp.py`

These currently contain overlapping concerns and should be separated by hunk
when preparing future PRs.

## Perfect-model standalone status

Progress made in this branch:

- added a historical ALADIN packaging mode in `build_dataset_pp.py`
  - `--include-train-hist`
  - `--historical-only`
- added training dataset override support
  - `bin/training/train.py --sample-dir`
  - `iriscc/hparams.py` sample-dir plumbing
- added standalone orchestration:
  - `bin/production/run_exp5_perfect_model.py`
- added Grace wrapper:
  - `bin/production/submit_exp5_perfect_model_grace.sh`

Current validation state:

- corrected scientific definition:
  - pseudo-truth `y` is the native/high-resolution RCM regridded to the ML
    target grid
  - predictor `x[1]` must be generated from the same native model by degrading
    through a configurable coarse bridge grid, then remapping to the ML target
    tensor
  - `ALADIN-BC/*gr_150km*_bc.nc` files are not acceptable perfect-model inputs
    because the tested files repeat the historical calendar sequence
- current implementation correction:
  - `build_dataset_pp.py` can now build perfect-model inputs from native RCM
    through `perfect_model_input_grid_source`
  - `perfect_model_rcm` currently sets that bridge to `gcm_cnrm_cm6_1`
  - `run_exp5_perfect_model.py` no longer uses `--corrected` for perfect-model
    sample generation
- smoke gate after correction:
  - `2000-01-01` and `2015-01-01` no longer have identical `x[1]`
  - `x[1]` mean absolute difference between those two dates is about `1.88 K`
  - degraded-input/native-target RMSE is about `1.75 K` on both smoke dates
- corrected full sample rebuild is running on Kraken as jobs `1884642-1884644`
  and `1884651-1884655`
- corrected full sample rebuild completed with `44195 / 44195` samples
- validation job `1884666` passed inventory, structure, and cross-period repeat
  checks
- statistics job `1884667` completed and wrote `statistics.json`
- corrected-data UNet and MiniUNet trainings are running:
  - UNet: `1884669`
  - MiniUNet: `1884684`
- prediction/comparison chunks are queued behind each training

- local smoke of the historical ALADIN packaging path succeeded
- Grace standalone historical dataset materialization succeeded
- Grace standalone statistics succeeded
- Grace 1-epoch training smoke succeeded
- Grace 30-epoch standalone training succeeded
- Grace standalone historical prediction for `unet_perfect_model_rcm` succeeded
- Grace ML daily/monthly metrics and VALUE metrics succeeded
- Kraken raw-RCM baseline daily/monthly metrics succeeded for the historical
  standalone perfect-model evaluation
- Kraken historical ML-vs-RCM comparison table was generated under
  `metrics/exp5/comparison_tables`
- Kraken future perfect-model samples were generated for `2015-2100` and
  validated for inventory, structure, boundary continuity, and plausible tas
  ranges
- Kraken future ML predictions were generated as six independent GPU chunks:
  - `2015-2029`
  - `2030-2044`
  - `2045-2059`
  - `2060-2074`
  - `2075-2089`
  - `2090-2100`
- Kraken future ML-vs-raw-RCM comparison table was generated under
  `metrics/exp5/comparison_tables`
- first scientific read of the future comparison:
  - ML mean bias against raw RCM evolves from about `+0.29 K` in `2015-2029`
    to about `-0.67 K` in `2090-2100`
  - ML warms by about `+2.49 K` between the first and last future windows,
    while raw RCM warms by about `+3.45 K`
  - window RMSE is about `3.0-3.3 K`
  - this suggests the trained ML path damps late-century RCM warming and should
    be treated as a scientific finding to investigate, not a technical failure
- RMSE diagnostic added:
  - `bin/evaluation/diagnose_perfect_model_rmse.py`
  - historical sampled all-domain `ML-y` RMSE is about `3.52 K`, versus raw
    `RCM-y` RMSE about `4.83 K`
  - historical sampled all-domain `ML-y` correlation is about `0.871`, versus
    raw `RCM-y` correlation about `0.769`
  - future `ML-raw RCM` RMSE stays near `3.0-3.4 K` after de-biasing, so the
    difference mostly reflects spatial/day-to-day structure and variability,
    not just the small mean offset
  - correction after scientific review: the previous future comparison was only
    a technical dry run because future samples did not contain pseudo-truth `y`
  - proper perfect-model setup is:
    - `x`: degraded/coarsened model input, currently bias-corrected/regridded
      RCM at `150 km`
    - `y`: native/high-resolution output from the same model regridded to the
      target grid, for historical and future
    - train/calibrate on historical `x -> y`
    - evaluate both historical and future predictions against `y`
  - implemented reusable config key `perfect_model_rcm`; the current source is
    ALADIN through `rcm_source='rcm_aladin'` and
    `perfect_model_target_source='rcm_aladin'`, but the experiment name is not
    ALADIN-specific
  - `build_dataset_pp.py` now supports `--perfect-model-target-source` so a
    native model source can be written as pseudo-truth `y` for all dates
  - one-day smokes passed:
    - `2000-01-01`: degraded-input vs native-target RMSE about `3.44 K`
    - `2015-01-01`: degraded-input vs native-target RMSE about `4.45 K`
  - full corrected `dataset_perfect_model_rcm` generation is running on Kraken
    in eight chunks

Operational notes from the future run:

- future raw RCM sample generation originally OOM-killed at `32G`
- chunked sample generation with `64G` completed; observed peak RSS was about
  `31-44G`
- shared xESMF weight files were fragile under concurrent access, so parallel
  preprocessing jobs used one private `IDOWNSCALE_REGRID_WEIGHTS_DIR` per Slurm
  chunk and `--skip-existing`
- future prediction parallelizes cleanly by independent date windows
- future comparison also parallelizes cleanly by the same date windows; six
  `prodshared` jobs completed in about `24-29 s`

## Kraken status

Kraken is now the primary runtime.

Validated on Kraken:

- direct local runtime on `kraken1.cluster`
- core x86 GPU environment:
  - `/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1`
- CPU imports:
  - `numpy`, `pandas`, `xarray`, `matplotlib`, `pyproj`, `cartopy`, `xesmf`
  - `torch`, `pytorch_lightning`, `torchmetrics`
  - `ibicus`, `monai`, `timm`
- GPU probe:
  - `torch 2.5.1`
  - `CUDA 12.4`
  - `cuda_available = True`
  - `device0 = NVIDIA A30`

Important Kraken note:

- `SBCK 1.4.2` is installed and validated in the Kraken env
- `SBCK.CDFt` import plus a tiny fit/predict smoke passed
- `bias_correction_sbck.py` wrapper passed on a tiny real GCM subset
- use `gpua30` by default
- use `rome` if memory becomes the bottleneck

Current migration follow-up:

- raw data and repo are present on Kraken
- Codex state was copied once; re-sync `~/.codex` again just before any future
  runtime/tunnel move if needed
- Kraken output paths now match the structured Calypso layout for the active
  `runs/`, `metrics/`, `datasets/`, `prediction/`, `graph/`, and `weights/`
  trees
- earlier flattened duplicate output directories were moved to a legacy backup
  instead of being deleted

Completed Kraken jobs on 2026-06-03:

- `1884136`: `pm_fut_2015_2029_p`
- `1884137`: `pm_fut_2030_2044_p`
- `1884138`: `pm_fut_2045_2059_p`
- `1884139`: `pm_fut_2060_2074_p`
- `1884140`: `pm_fut_2075_2089_p`
- `1884141`: `pm_fut_2090_2100_p`

All six jobs ran on `prodshared` with `--mem=64G`, private regrid-weight
directories, and `--skip-existing`.

Perfect-model future sample validation:

- report:
  - `/scratch/globc/page/output/metrics/exp5/validation/perfect_model_samples_exp5_rcm_20000101_21001231.md`
- inventory status: `ok`
- structure status: `ok`
- expected samples for `2000-01-01` through `2100-12-31`: `36890`
- actual samples: `36890`
- missing dates: `0`
- extra dates: `0`
- boundary-day value distributions around `2014-12-31` / `2015-01-01`
  are continuous enough to clear the next GPU prediction step

Path-default cleanup:

- Python modules under `iriscc/` and `bin/` no longer carry hidden
  `/gpfs-calypso/...` defaults
- default output root in `settings.py` is repo-local unless explicitly
  configured

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
   - treat strict byte-for-byte parity as optional if the archive reference is
     shown to be internally inconsistent
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
  - `/gpfs-calypso/scratch/globc/page/output/validation_20260528`
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
  - as of 2026-06-01 it now exports `ESMFMKFILE` by default so `xesmf` works with the repaired Grace venv without extra operator setup
- Reusable `globc` CPU helper:
  - `bin/production/run_in_globc_cpu_env.sh` runs the x86 CPU fallback env while clearing inherited `PYTHONHOME`/`PYTHONPATH`
  - use this when reproducing `ibicus` issues on the login/CPU side because direct `conda run` can fail with `ModuleNotFoundError: encodings`
- Grace venv repair completed on 2026-06-01:
  - repaired `/scratch/globc/page/idownscale_envs/production_final_v22_312`
  - verified on-arm imports for `numpy`, `scipy`, `xarray`, `dask`, `ibicus`, `xesmf`, `torch`, `pytorch_lightning`, `tensorboard`, and `timm`
  - `pip check` now reports `No broken requirements found`
  - verified again through the default wrapper path with probe job `272675`

### Known limits

- Production BC methods are `ibicus_cdft` and `sbck_cdft`; other methods still raise explicitly.
- `sbck_cdft` additionally requires the `SBCK` Python package to be present in the Grace environment.
- Runtime validation still depends on the target Grace environment having `cartopy`, `xesmf`, `torch`, and the expected data roots available.
- Strict ALADIN archival parity is no longer treated as a reliable success
  criterion on its own:
  - raw packaged RCM samples now match Zoé's reference structurally
  - historical BC bundles now match essentially exactly with native ALADIN bounds
  - corrected-sample archive contains at least one proven inconsistency:
    `sample_19800101.npz` stores `y` in Celsius instead of Kelvin
  - future BC bundle archive `bc_test_future_rcm.npz` appears internally mixed:
    `2015-2069` differs systematically by about `0.10 K` mean abs diff, while
    `2070+` matches the regenerated native-bounds path essentially exactly
  - `/scratch/.../bc_test_future_rcm.npz` and `/archive2/.../bc_test_future_rcm.npz`
    are byte-identical mirrors, so the inconsistency is preserved in both archive roots
  - the regenerated ALADIN/RCM workflow is therefore treated as scientifically
    validated but not expected to reproduce every inconsistent archive artifact

## Perfect-model next step

The next engineering target after the Grace parity campaign is to isolate the
ALADIN perfect-model workflow as a standalone production path:

- upscale ALADIN from its fine curvilinear grid onto the coarse `gr_150km`
  bridge grid
- downscale back to the fine target grid through the ML pipeline
- compare ML output directly against the fine-resolution ALADIN reference on the
  historical period

This should make ML-vs-RCM comparison reproducible without depending on the
broader archive-parity framing.

## Date-boundary hardening

To stay compatible with future archives such as CMIP7, do not assume that the
historical/scenario cutoff is always `2014-12-31` / `2015-01-01`.

Current branch status:

- workflow date propagation was fixed separately and merged to `master`
- this branch additionally removed remaining runtime assumptions from:
  - `run_exp5_workflow.py`
  - `run_exp5_perfect_model.py`
  - `predict_loop.py`
  - `iriscc/datautils.py`
  - `iriscc/settings.py`
- source-specific scenario-start overrides are now available for:
  - GCM
  - CORDEX
  - ALADIN/RCM

Remaining fixed dates in the repo are mostly current-exp5 configuration or
plotting/report helpers, not hidden workflow defaults.

## Master fix PR in flight

A separate `master`-oriented branch was prepared from `origin/master` in a clean
worktree:

- worktree:
  - `/scratch/globc/page/idownscale_master_boundary_fix`
- branch:
  - `fix/date-boundary-configurable`
- pushed commit:
  - `e4d2823` `Remove hardcoded date cutoffs from workflow paths`
- PR URL:
  - `https://github.com/cerfacs-globc/idownscale/pull/new/fix/date-boundary-configurable`

That branch includes:

- explicit date args in docs/runbooks
- workflow/runtime date-boundary cleanup
- updates to the downstream scripts called by `run_exp5_full.sh`

Important handoff note:

- there is an extra **local-only** uncommitted edit in
  `/scratch/globc/page/idownscale_master_boundary_fix/iriscc/settings.py`
  exploring a broader “no hardcoded model/reanalysis identity” refactor
- this is intentionally **not** part of the pushed `e4d2823` commit
- recommendation remains:
  - keep the date-boundary PR small and reviewable
  - open a separate follow-up PR for full source/model/grid/projection
    generalization

## Broader generalization requirement

The eventual framework should avoid hardcoding:

- reanalysis name
- model name
- grid/projection identity
- historical/future cutoff dates
- filename conventions tied to one source family

This broader requirement should be tracked as the next generalization PR rather
than folded into the small master hotfix.

## Parallel execution rule

At the end of the session, the intended sequencing became:

1. finish the narrow `master` fix branch
2. define or prepare the broader “nothing hardcoded” follow-up
3. let ALADIN/RCM stabilization and perfect-model execution continue in parallel

Items 1 and 2 should progress while item 3 is running.

## Perfect-model Kraken checkpoint

The corrected perfect-model run is no longer a Grace handoff item. It completed
on Kraken with the scientifically valid pseudo-reality construction:

- coarse predictor `x`:
  - native RCM degraded to the configurable coarse bridge grid
  - then remapped to the ML target grid
- pseudo-truth `y`:
  - native RCM remapped to the ML target grid
- historical training:
  - degraded historical RCM input to native historical RCM target
- evaluation:
  - historical and future ML predictions against native RCM pseudo-truth

Completed artifacts:

- dataset:
  - `/scratch/globc/page/output/datasets/dataset_bc/dataset_perfect_model_rcm`
  - `44195 / 44195` samples
- validation:
  - job `1884666`
  - inventory, structure, and cross-period repeat checks all `ok`
  - report:
    - `/scratch/globc/page/output/metrics/perfect_model_rcm/validation/perfect_model_samples_perfect_model_rcm_rcm_19800101_21001231.md`
- statistics:
  - job `1884667`
  - report:
    - `/scratch/globc/page/output/datasets/dataset_bc/dataset_perfect_model_rcm/statistics.json`
- combined ML-vs-RCM table:
  - `/scratch/globc/page/output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.md`

Current interpretation:

- UNet is the strongest valid benchmark:
  - bias stays between `+0.044 K` and `+0.071 K`
  - RMSE stays around `0.47-0.49 K`
  - coarse-input RMSE is about `1.68-1.77 K`
- MiniUNet is useful as a secondary architecture result:
  - RMSE is good at about `0.52-0.57 K`
  - bias is systematically cold at about `-0.24 K` to `-0.29 K`, above the
    current `0.10 K` tolerance

Next implementation target:

- package this as a standalone perfect-model experiment that does not encode
  ALADIN in the public concept name
- keep the source configurable so the same workflow can be reused with another
  RCM, variable, ML method, or coarse bridge grid
- preserve validation gates between phases so production runs fail early if
  structure, inventory, cross-period repeats, or basic value distributions are
  wrong

Implementation note:

- `bin/production/run_exp5_perfect_model.py` now imports `CONFIG` correctly.
- The launcher accepts `--sample-dir` so a Kraken/Grace/local run can point at
  the intended perfect-model dataset explicitly.
- This avoids a silent mismatch between repo-local default outputs and the
  production output tree when output-root environment variables are not set.

Active follow-up:

- MiniSwinUNETR is being trained on the corrected dataset as a third
  architecture comparison.
- Training job:
  - `1884757`
- Dependent prediction/comparison chain:
  - predictions `1884758`, `1884760`, `1884762`, `1884764`, `1884766`,
    `1884768`, `1884770`
  - comparisons `1884759`, `1884761`, `1884763`, `1884765`, `1884767`,
    `1884769`, `1884771`
- The first MiniSwin launch `1884742` was canceled because it inherited a
  repo-local output root; the relaunched chain sets `/scratch/globc/page/output`
  explicitly.
- The clean MiniSwin run `1884757` was canceled at the epoch-6 gate:
  - validation loss was still `234.2`
  - successful UNet/MiniUNET were already near `46` and `37` by the same epoch
  - dependent prediction/comparison jobs were canceled because this architecture
    configuration was not worth evaluating scientifically
- A MONAI UNet comparison was launched next:
  - training job `1884774`
  - prediction jobs `1884775`, `1884777`, `1884779`, `1884781`, `1884783`,
    `1884785`, `1884787`
  - comparison jobs `1884776`, `1884778`, `1884780`, `1884782`, `1884784`,
    `1884786`, `1884788`
  - canceled at the epoch-6 gate because validation loss was still `226.3`,
    far from the successful UNet/MiniUNet regime
- Two standard UNet robustness replicates are now active:
  - `1884803` / `unet_seed2_perfect_model_rcm`
  - `1884819` / `unet_rep3_perfect_model_rcm`
  - both have dependent prediction and comparison chains across the historical
    and six future windows
- Current gate:
  - let each replicate continue only if the epoch-6 validation curve is close
    to the successful UNet behavior
  - aggregate `unet_perfect_model_rcm`, `miniunet_perfect_model_rcm`,
    `unet_seed2_perfect_model_rcm`, and `unet_rep3_perfect_model_rcm` after
    the comparison chunks complete

Completed follow-up:

- The seed2 and rep3 UNet replicates both passed the epoch gate, but seed2
  finished with a cold bias outside the `0.10 K` working tolerance.
- A standard UNet trained with `--output-norm` was added as a targeted
  optimization experiment.
- `predict_loop.py` was fixed to de-normalize output-normalized predictions
  before writing NetCDF; without this, predictions stayed in normalized units.
- The current combined perfect-model result is:
  - table:
    `/scratch/globc/page/output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.md`
  - best candidate:
    `unet_outputnorm_perfect_model_rcm`
  - output-normalized UNet bias:
    `-0.016` to `-0.005 K`
  - output-normalized UNet RMSE:
    about `0.452-0.477 K`
  - raw degraded-input RMSE:
    about `1.68-1.77 K`
- Operational lesson:
  - for this standalone perfect-model workflow, training submissions should use
    `--skip-test`; the separate prediction/comparison chain is the scientific
    evaluation and avoids a long redundant post-fit test pass
- Workflow/productization follow-up completed:
  - `run_exp5_perfect_model.py` now exposes explicit validation and comparison
    steps instead of leaving them as standalone manual scripts:
    - `validate_train_dataset`
    - `validate_eval_dataset`
    - `compare_predictions`
    - `aggregate_comparison`
    - `plot_score_comparison`
    - `plot_distribution`
  - the standalone workflow now uses `--skip-test` for training by default
  - a Kraken submitter was added:
    - `bin/production/submit_exp5_perfect_model_kraken.sh`
  - perfect-model diagnostics now carry more provenance:
    - variable name
    - variable label
    - unit
    - selected source-role metadata copied from prediction NetCDF attrs
  - comparison tables now provide generic metric columns in addition to legacy
    `_K` aliases, so the same machinery can be reused beyond temperature

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
    - `/gpfs-calypso/scratch/globc/page/output/metrics/exp5/bc_validation_summary_exp5_gcm.csv`
    - `/gpfs-calypso/scratch/globc/page/graphs/metrics/exp5/bc_validation_pdf_exp5_gcm.png`
  - note:
    - current default GCM corrected NetCDFs only overlap one smoke day with the full BC bundles, so these numbers validate the method and not yet a full-period scientific GCM certification
- ALADIN/RCM BC dataset performance fix validated on Grace:
  - job `267491`
  - result:
    - optimized `build_dataset_bc.py` now processes by source-file batch instead of one day at a time
    - `--test` smoke for `simu=rcm` completed in about 10 seconds
  - status:
    - this is the active path for the new full ALADIN parity rerun

Perfect-model BC follow-up lesson, 2026-06-04:

- the earlier poor BC-baseline diagnosis for `perfect_model_rcm` was caused by
  a configuration mismatch:
  degraded ALADIN was being bias-corrected against `ERA5`, then evaluated
  against native `ALADIN`
- the scientific fix is to use
  `CONFIG['perfect_model_rcm']['bc_reanalysis_source'] = 'rcm_aladin'`
- the first corrected rerun (`1887337`) still failed, but with a purely
  technical `OUT_OF_MEMORY` during corrected sample packaging after the BC
  NetCDFs had already been written
- technical cause:
  - corrected sample packaging reopened `orog_file` for every day
  - per-day regridded datasets from `reformat_as_target(...)` were not being
    closed promptly
- mitigation now implemented in both `bias_correction_ibicus.py` and
  `bias_correction_sbck.py`:
  - load orography once
  - package corrected samples through a shared helper
  - close each one-day regridded dataset immediately
  - close corrected historical/future datasets after packaging
- carry this rule forward:
  - when `bc_apply` OOMs after corrected BC files already exist, first inspect
    sample packaging memory behavior before questioning the debiasing method

## Active Grace jobs

- Full ALADIN/RCM BC rerun in isolated parity root:
  - current job `272669`
  - output root:
    - `/gpfs-calypso/scratch/globc/page/output/validation_20260528/rcm_parity_full`
  - current launch path:
    - `bin/production/submit_rcm_parity_grace.sh`
  - note:
    - this submitter currently uses the mixed fallback wrapper for safety while the repaired default env is being revalidated in parallel
- Dependent ALADIN/RCM scientific + archive parity validation:
  - current job `272670`
  - starts after successful completion of job `272669`
- Earlier failure context from the first parity attempt on 2026-06-01:
  - `267492` failed in `bc_apply` inside `bin/preprocessing/bias_correction_ibicus.py`
  - failing call: `CDFt.from_variable(...).apply(...)` on the full ALADIN/RCM parity dataset
  - error raised by `ibicus`:
    - time-information dimensions reported as inconsistent with `obs/cm_hist/cm_future`
  - stale dependent validation job:
    - `268942` is `PENDING` with `DependencyNeverSatisfied`
  - do not simply resubmit the same dependency chain until the `ibicus` caller or env path is fixed and locally reproduced on CPU or Grace

## Perfect-model BC recovery completion, 2026-06-05

Outcome:

- bounded Kraken recovery chain completed
- corrected BC baseline was regenerated
- perfect-model ML rerun using the BC-aligned workflow path completed
- comparison, PDF, and climate-signal diagnostics were regenerated successfully

Successful jobs:

- prep / corrected BC rebuild:
  - `1888048`
- late-century ML jobs:
  - `1888548` `pm_bc8_uon`
  - `1888549` `pm_bc8_unet`
  - `1888550` `pm_bc8_rep3`
  - `1888551` `pm_bc8_mini`
  - `1888552` `pm_bc8_seed2`
- post-processing:
  - `1888553` `pm_bc8_post`

Main bugs discovered and fixed:

1. Perfect-model BC used the wrong reference
   - fixed by setting
     `CONFIG['perfect_model_rcm']['bc_reanalysis_source'] = 'rcm_aladin'`

2. Corrected perfect-model sample packaging rebuilt `y` through the wrong path
   - fixed in `bias_correction_ibicus.py` and `bias_correction_sbck.py`

3. Corrected sample packaging OOM
   - fixed by loading orography once and closing one-day regridded datasets
     immediately

4. Corrected eval dataset contamination
   - fixed in `build_dataset_pp.py` so `--corrected` no longer implies
     `train_hist`

5. Future sample-dir resolution bugs
   - fixed in `predict_loop.py`
   - fixed again downstream in:
     - `plot_perfect_model_distribution_pdf.py`
     - `compare_perfect_model_climate_signal.py`

6. Recovery relaunch shape initially assumed broader future coverage than was
   materialized on disk
   - corrected final relaunch limited to `20900101_21001231`

Final scientific checkpoint:

- combined windows currently available:
  - `20000101_20141231`
  - `20900101_21001231`
- late-century raw coarse input and BC baseline:
  - bias `0.230943 K`
  - RMSE `1.760735 K`
- late-century best ML result:
  - `UNet + output norm`
  - bias `-0.037046 K`
  - RMSE `0.452707 K`
- climate-signal truth mean:
  - `3.619777 K`
- climate-signal RMSE:
  - raw coarse input: `0.160099 K`
  - BC baseline: `0.160117 K`
  - `UNet + output norm`: `0.079900 K`
  - `MiniUNet`: `0.075264 K`

Final technical checkpoint:

- no active SLURM jobs remain for this bounded workflow
- regenerated outputs:
  - combined table:
    - `/scratch/globc/page/output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.csv`
  - score plot:
    - `/scratch/globc/page/graphs/metrics/perfect_model_rcm/perfect_model_method_comparison_perfect_model_rcm_rcm.png`
  - PDF plot:
    - `/scratch/globc/page/graphs/metrics/perfect_model_rcm/perfect_model_distribution_pdf_perfect_model_rcm_rcm_tas.png`
  - climate-signal table:
    - `/scratch/globc/page/output/metrics/perfect_model_rcm/comparison_tables/perfect_model_climate_signal_perfect_model_rcm_rcm_20000101_20141231_vs_20900101_21001231.csv`
