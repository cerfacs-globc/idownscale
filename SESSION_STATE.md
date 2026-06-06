# Session State

Last snapshot: 2026-06-03 CEST

## Current 2026-06-03 Kraken state

Kraken is the active runtime now, not Grace.

Kraken default project environment:

- interpreter: `/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1/bin/python`
- activation: `source /scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1/bin/activate`
- use this as the first-choice runtime for Kraken checks, preprocessing,
  evaluation, and plotting unless a job script explicitly pins another env

Latest correction, 2026-06-03 late morning:

- the first full `perfect_model_rcm` dataset build was scientifically invalid:
  the degraded/coarse input channel `x[1]` repeated by day index across
  historical and future periods
- root cause:
  - the workflow used `--corrected`
  - that routed perfect-model input through `rawdata/rcm/ALADIN-BC/*gr_150km*_bc.nc`
  - those files repeat the historical calendar sequence and are not valid
    future pseudo-reality inputs
- code correction:
  - `build_dataset_pp.py` now supports perfect-model input generation from the
    native model source through a configurable coarse bridge grid
  - current config uses `perfect_model_input_grid_source='gcm_cnrm_cm6_1'`
    only as the coarse grid source for this experiment
  - `run_exp5_perfect_model.py` no longer passes `--corrected` for
    perfect-model sample generation
- smoke gate passed:
  - `2000-01-01` x/y RMSE about `1.75 K`
  - `2015-01-01` x/y RMSE about `1.75 K`
  - `x[1]` mean absolute difference between the two dates is about `1.88 K`
  - this replaces the invalid zero-difference behavior
- perfect-model BC follow-up, 2026-06-04:
  - the earlier “BC baseline is bad” interpretation was traced to a
    configuration mismatch:
    degraded ALADIN was bias-corrected against `ERA5`, then evaluated against
    native `ALADIN`
  - `CONFIG['perfect_model_rcm']['bc_reanalysis_source']` is now
    `rcm_aladin`
  - corrected-sample writers in `bias_correction_ibicus.py` and
    `bias_correction_sbck.py` now build perfect-model pseudo-truth through the
    same native-model path as `build_dataset_pp.py`
  - first corrected rerun job `1887337` failed with `OUT_OF_MEMORY` after
    `46m53s`, but only after corrected BC NetCDFs had already been written
  - diagnosed technical cause:
    - corrected sample packaging reopened `orog_file` for every date
    - per-day `reformat_as_target(...)` datasets were not being closed
      promptly
  - mitigation now in code:
    - load orography once per run
    - materialize corrected samples through a shared helper
    - close each regridded day dataset immediately after extracting values
    - close corrected historical/future datasets after packaging
  - operational rule:
    - if `bc_apply` OOMs after corrected NetCDFs exist, treat corrected sample
      packaging as the likely culprit before blaming the BC algorithm
- invalid dataset archived under:
  `/scratch/globc/page/idownscale_output/datasets/dataset_bc/dataset_perfect_model_rcm_bad_repeating_20260603_*`
- corrected full rebuild is running:
  - `1884642`: `1980-1999` on `icelake`
  - `1884643`: `2000-2014` on `icelake`
  - `1884644`: `2015-2029` on `icelake`
  - `1884651`: `2030-2044` on `gpua30`
  - `1884652`: `2045-2059` on `gpua30`
  - `1884653`: `2060-2074` on `gpua30`
  - `1884654`: `2075-2089` on `gpua30`
  - `1884655`: `2090-2100` on `gpua30`
- corrected rebuild completed:
  - `44195 / 44195` samples present
  - peak RSS by chunk was about `35-47 GB`
- validation job `1884666` passed:
  - inventory `ok`
  - structure `ok`
  - cross-period repeat check `ok`
  - validation report:
    `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/validation/perfect_model_samples_perfect_model_rcm_rcm_19800101_21001231.md`
- stats job `1884667` completed in `19:41`:
  - wrote `/scratch/globc/page/idownscale_output/datasets/dataset_bc/dataset_perfect_model_rcm/statistics.json`
  - degraded input mean/std: about `283.57 K / 6.70 K`
  - native target mean/std: about `283.22 K / 7.05 K`
- active corrected-data training:
  - UNet job `1884669`, test name `unet_perfect_model_rcm`
  - MiniUNet job `1884684`, test name `miniunet_perfect_model_rcm`
  - both are on `gpua30`, dependency-released after stats
- prediction/comparison chains are already queued behind each training:
  - UNet predictions `1884670,1884672,1884674,1884676,1884678,1884680,1884682`
  - UNet comparisons `1884671,1884673,1884675,1884677,1884679,1884681,1884683`
  - MiniUNet predictions `1884685,1884687,1884689,1884691,1884693,1884695,1884697`
  - MiniUNet comparisons `1884686,1884688,1884690,1884692,1884694,1884696,1884698`

Current working rule:

1. verify implementation/runtime correctness
2. verify structural data integrity
3. verify scientific data values, including CDF/quantile behavior
4. only then launch expensive downstream GPU prediction or full metrics

Workflow adjustment to carry forward:

- do not run phase 1 through phase 6 as one opaque block and discover a phase-1
  problem at the end
- add explicit gates after phases in `run_exp5_workflow.py` or companion
  `verify_*` scripts
- each gate should combine cheap structural checks with value/CDF checks where
  the phase changes scientific data
- the existing final validation/report remains useful, but it should summarize
  phase gates that already passed

This maps to the usual numerical-simulation V&V/UQ framing:

- verification: did we implement and run the calculation correctly?
- validation: are outputs credible for the scientific use case?
- uncertainty/sensitivity: what residual differences are acceptable and why?

Perfect-model future sample build:

- six Kraken `prodshared` chunks completed on 2026-06-03:
  - `1884136`: `2015-2029`
  - `1884137`: `2030-2044`
  - `1884138`: `2045-2059`
  - `1884139`: `2060-2074`
  - `1884140`: `2075-2089`
  - `1884141`: `2090-2100`
- all used `--mem=64G`, private `IDOWNSCALE_REGRID_WEIGHTS_DIR`, and
  `--skip-existing`
- `32G` was not enough for the original monolithic run
- observed peak RSS was about `31-44G`, so `64G` is a reasonable request for
  this chunked preprocessing
- dataset inventory validation passed:
  - expected samples for `2000-01-01` through `2100-12-31`: `36890`
  - actual samples: `36890`
  - missing dates: `0`
  - extra dates: `0`
- structural validation passed:
  - historical samples contain `x` and `y`, both on `(64, 64)`
  - future samples contain `x` only, on `(64, 64)`
  - `x` has two channels: orography and remapped RCM tas
- first value-distribution validation passed basic plausibility checks:
  - historical sampled `x` tas mean about `278.47 K`, target `y` mean about
    `278.24 K`
  - historical sampled `x-y` mean about `0.23 K`
  - boundary days around `2014-12-31` / `2015-01-01` are continuous in mean and
    quantiles
  - future sampled `x` values stay in plausible Kelvin ranges through 2100

SBCK on Kraken:

- installed in `/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1`
- import validation succeeded with `SBCK 1.4.2`
- `CDFt` is present and a tiny column-vector fit/predict smoke returned finite
  outputs

Future perfect-model prediction and comparison:

- six Kraken `gpua30` prediction chunks completed on 2026-06-03:
  - `1884149`: `2015-2029`
  - `1884150`: `2030-2044`
  - `1884151`: `2045-2059`
  - `1884152`: `2060-2074`
  - `1884153`: `2075-2089`
  - `1884154`: `2090-2100`
- all completed in about `3.5-4 min` each with peak RSS around `1.2G`
- prediction output files are under:
  `/scratch/globc/page/idownscale_output/prediction`
- six Kraken `prodshared` comparison chunks completed on 2026-06-03:
  - `1884157`: `2015-2029`
  - `1884158`: `2030-2044`
  - `1884159`: `2045-2059`
  - `1884160`: `2060-2074`
  - `1884161`: `2075-2089`
  - `1884162`: `2090-2100`
- all comparison chunks completed in about `24-29 s`
- merged future comparison outputs:
  - `/scratch/globc/page/idownscale_output/metrics/exp5/comparison_tables/perfect_model_future_ml_vs_rcm_exp5_unet_perfect_model_rcm_rcm.csv`
  - `/scratch/globc/page/idownscale_output/metrics/exp5/comparison_tables/perfect_model_future_ml_vs_rcm_exp5_unet_perfect_model_rcm_rcm.md`
- first scientific read:
  - ML mean bias against raw RCM moves from about `+0.29 K` in `2015-2029`
    to about `-0.67 K` in `2090-2100`
  - ML warming between first and last future windows is about `+2.49 K`
  - raw RCM warming between first and last future windows is about `+3.45 K`
  - window RMSE is about `3.0-3.3 K`
- operational lesson:
  - future prediction is safe to parallelize by independent date windows
  - future comparison is safe to parallelize by independent date windows
  - regridding sample generation is also parallelizable, but each job needs a
    private `IDOWNSCALE_REGRID_WEIGHTS_DIR`

Perfect-model RMSE diagnostic:

- diagnostic script:
  `bin/evaluation/diagnose_perfect_model_rmse.py`
- Slurm job `1884174` completed on Kraken `prodshared` in `44 s`
- outputs:
  - `/scratch/globc/page/idownscale_output/metrics/exp5/diagnostics/perfect_model_rmse_diagnostics_exp5_unet_perfect_model_rcm_rcm.csv`
  - `/scratch/globc/page/idownscale_output/metrics/exp5/diagnostics/perfect_model_rmse_diagnostics_exp5_unet_perfect_model_rcm_rcm.md`
- historical sampled all-domain result:
  - `ML-y` RMSE about `3.52 K`, correlation about `0.871`
  - `raw RCM-y` RMSE about `4.83 K`, correlation about `0.769`
  - therefore the ML path improves over raw RCM against the withheld ALADIN-like
    target in the historical perfect-model evaluation
- future `ML-raw RCM` result:
  - RMSE remains about `3.0-3.4 K` after removing mean bias, so it is mostly
    spatial/day-to-day structure and variability difference, not a simple
    domain-mean offset
  - ML standard deviation is lower than raw RCM in all windows, consistent with
    damped variability and late-century warming
Scientific correction to the perfect-model workflow:

- the previous future comparison was a technical dry run, not the proper
  perfect-model evaluation
- proper pseudo-reality/perfect-model setup:
  - `x` is the degraded/coarsened model field, currently ALADIN at `150 km`
  - `y` is the native/high-resolution output from the same model, regridded to
    the target grid
  - `y` must exist for both historical and future dates
  - skill is evaluated against `y` in both historical and future
- code changes made:
  - added reusable config `perfect_model_rcm`
  - kept ALADIN only as the current configured source:
    `rcm_source='rcm_aladin'` and
    `perfect_model_target_source='rcm_aladin'`
  - added `build_dataset_pp.py --perfect-model-target-source`
  - updated standalone runner defaults to `perfect_model_rcm` and
    `unet_perfect_model_rcm`
- validation smokes:
  - `perfect_model_rcm` future one-day sample for `2015-01-01` contains `x`
    and `y`, both on `(64, 64)`
  - `2015-01-01` degraded-input/native-target RMSE is about `4.45 K`
  - `2000-01-01` degraded-input/native-target RMSE is about `3.44 K`
- full corrected dataset rebuild launched on Kraken:
  - output: `/scratch/globc/page/idownscale_output/datasets/dataset_bc/dataset_perfect_model_rcm`
  - jobs: `1884176` through `1884183`
  - windows:
    - `1980-1999`
    - `2000-2014`
    - `2015-2029`
    - `2030-2044`
    - `2045-2059`
    - `2060-2074`
    - `2075-2089`
    - `2090-2100`

## Current branch

`grace-bc-parity-20260528`

## Current objective

Resume the ALADIN archival parity/validation work for the generalized exp5-style
workflow on Grace, using Zoé's historical outputs as the reference.

Current conclusion:

- the regenerated ALADIN/RCM workflow is technically healthy and scientifically
  close enough for validation use
- strict archive parity is no longer treated as a reliable target because the
  archive itself contains inconsistencies
- the next concrete engineering target is to expose the ALADIN perfect-model
  workflow as a standalone run for ML-vs-RCM comparison
- merge strategy has changed:
  - date-propagation fixes were split out and merged independently
  - broader source-generalization work should be split later, after ALADIN/RCM
    support is considered stable enough against Zoé's reference with tolerance
  - perfect-model standalone support should be extracted in a separate branch/PR

## Working split map

Use the current branch as a staging area, but classify changes into these
buckets before preparing future PRs.

### 1. ALADIN/RCM stabilization

Primary goal:
- make ALADIN/RCM technically correct
- validate against Zoé's archive with documented tolerances

Mostly in this bucket:
- `iriscc/datautils.py`
- `bin/preprocessing/build_dataset_bc.py`
- `bin/preprocessing/bias_correction_ibicus.py`
- `bin/preprocessing/bias_correction_sbck.py`
- `bin/evaluation/validate_bc_outputs.py`
- `bin/production/probe_rcm_future_parity_modes.py`

Notes:
- this bucket includes the native ALADIN bounds fix, curvilinear-grid handling,
  IBICUS location-wise workaround, and the BC validation tooling
- some helper changes in `iriscc/settings.py` are currently supporting this
  bucket, but the full source-catalog generalization is broader than what this
  stabilization bucket should eventually merge

### 2. Generalization framework

Primary goal:
- make workflow/source selection config-driven across reanalyses, models, and
  target datasets

Mostly in this bucket:
- `iriscc/settings.py`
- `bin/production/run_exp5_workflow.py`
- `bin/production/run_exp5_workflow_grace.sh`
- `bin/production/submit_exp5_workflow_grace.sh`
- `bin/production/run_in_grace_env.sh`

Likely also part of this bucket:
- parts of `bin/preprocessing/build_dataset_pp.py`

Notes:
- this includes `SOURCE_CATALOG`, direct-source routing, support for `era6`,
  `cerra`, `cordex`, `custom_obs`, and generalized workflow dispatch
- this should be merged only after ALADIN/RCM stabilization is trusted enough
  not to hide regressions behind generalized plumbing

### 3. Perfect-model standalone support

Primary goal:
- run the ALADIN perfect-model path on its own for ML-vs-RCM comparison

Mostly in this bucket:
- parts of `bin/preprocessing/build_dataset_pp.py`
- `bin/production/run_exp5_perfect_model.py`
- `bin/production/submit_exp5_perfect_model_grace.sh`

Notes:
- this is not just generic infrastructure; it is a separate user-facing
  capability and should be split from the broader source-generalization PR

### Mixed files needing hunk-level separation later

- `iriscc/settings.py`
  - ALADIN/RCM support currently depends on some new helpers here, but the file
    also contains broader multi-source generalization
- `bin/production/run_exp5_workflow.py`
  - currently mixes generalized source dispatch with ALADIN/RCM operational
    support
- `bin/preprocessing/build_dataset_pp.py`
  - currently mixes generalized simulation packaging and the beginnings of the
  perfect-model-support path

## Perfect-model progress

Implemented in this branch:

- `bin/preprocessing/build_dataset_pp.py`
  - `--include-train-hist` to package the `1980-1999` historical train window
  - `--historical-only` to keep a standalone run focused on historical ALADIN
    perfect-model evaluation
- `bin/training/train.py`
  - `--sample-dir` override so training can target a standalone perfect-model
    dataset instead of the default `CONFIG[exp]['dataset']`
- `iriscc/hparams.py`
  - corresponding `sample_dir` override plumbing
- `bin/production/run_exp5_perfect_model.py`
  - standalone orchestration for:
    - historical ALADIN training dataset build
    - historical ALADIN eval-dataset build
    - stats
    - train
    - ML historical prediction/metrics
    - raw-RCM baseline metrics
- `bin/production/submit_exp5_perfect_model_grace.sh`
  - Grace wrapper for the standalone perfect-model workflow

Smoke validation completed:

- local CPU-env smoke:
  - `bash bin/production/run_in_globc_cpu_env.sh bin/preprocessing/build_dataset_pp.py --exp exp5 --simu rcm --include-train-hist --historical-only --test --output_dir scratch/perfect_model_smoke`
  - produced:
    - `scratch/perfect_model_smoke/sample_19800101.npz`
    - `scratch/perfect_model_smoke/sample_20000101.npz`
  - both contain `x` and `y` with shapes `(2, 64, 64)` and `(1, 64, 64)`

Current live job:

- Grace job `273087`
  - submitted on 2026-06-01
  - steps: `build_train_dataset,build_eval_dataset,stats`
  - purpose: materialize the standalone ALADIN perfect-model historical dataset
    and compute its statistics before attempting a full training run

## Resume checklist

1. Read [doc/OBJECTIVE_PLAN.md](/scratch/globc/page/idownscale_rerun/doc/OBJECTIVE_PLAN.md:1), especially `Current objective`, `Current task status`, and `Recovery context`.
2. Inspect the latest local diffs before changing code:
   `git status --short`
3. Focus first on parity-validation files and source-resolution paths, especially:
   - `iriscc/settings.py`
   - `iriscc/datautils.py`
   - `bin/preprocessing/build_dataset_bc.py`
   - `bin/preprocessing/build_dataset_pp.py`
   - `bin/production/run_exp5_workflow.py`

## Suggested next commands

```bash
bash bin/production/snapshot_session_state.sh
git status --short
sed -n '1,180p' doc/OBJECTIVE_PLAN.md
```

Grace parity relaunch:

```bash
bash bin/production/submit_rcm_parity_grace.sh
```

CPU-side reproduction helper:

```bash
bash bin/production/run_in_globc_cpu_env.sh -c "import xarray, xesmf, ibicus; print('ok')"
```

## Notes for next chat

- Conversation history may be unreliable in Antigravity/remote sessions, so use
  this file plus `doc/OBJECTIVE_PLAN.md` as the recovery source of truth.
- Do not revert unrelated user changes in the dirty worktree.
- Before running long Grace jobs, sanity-check dry-run and smoke-test commands
  from `doc/OBJECTIVE_PLAN.md`.
- Calypso login shells may inherit a base-conda `PYTHONHOME` that breaks the
  `globc_cpu_py312_v2` env; use `bin/production/run_in_globc_cpu_env.sh`.
- Grace job `267492` failed in `bias_correction_ibicus.py` with an `ibicus`
  time-dimension mismatch; `268942` is only a stale blocked dependent job.
- Grace default env `/scratch/globc/page/idownscale_envs/production_final_v22_312`
  was repaired on 2026-06-01 and revalidated on-arm.
- Default Grace wrappers now export `ESMFMKFILE`, and wrapper probe job
  `272675` confirmed the repaired env imports cleanly.
- Native-bounds `conservative_normed` remap for ALADIN now gives essentially
  exact historical BC-bundle parity and exact corrected-sample `y` parity
  against the repaired overlay reference.
- Remaining discrepancy is confined to `bc_test_future_rcm.npz` for
  `2015-01-01` through `2069-12-31`; from `2070-01-01` onward the future BC
  bundle matches Zoé's archive essentially exactly.
- The early/mid-future residual is about `0.10 K` mean abs diff with a stable
  spatial bias pattern. It is not explained by:
  - raw-file drift: ALADIN raw files in this tree and `/archive2` are byte-identical
  - a one-day time shift
  - bilinear remapping
  - the pre-speedup legacy per-day loader path
- Grace probe jobs:
  - `273050`: future remap-mode probe; native-bounds conservative is by far the
    closest mode to archive on problematic dates
  - `273056`: ruled out the old final-domain cropped target path as an archive
    explanation because it changes bundle shape
  - `273057`: ruled out the pre-speedup legacy per-day path as an archive
    explanation
- Working hypothesis: Zoé's `bc_test_future_rcm.npz` archive is internally
  mixed or partially regenerated with an older intermediate workflow for
  `2015-2069`, while `2070+` reflects the current native-bounds path.
- Searches restricted to `/scratch/globc/garcia/idownscale` and
  `/archive2/globc/garcia/garcia_copy_scratch/idownscale` found no alternate
  `bc_test_future_rcm.npz` and no useful provenance notes or generation
  scripts; both archive roots contain the same byte-identical bundle.

## Working tree snapshot

<!-- SESSION_STATE_AUTOGEN_START -->
Generated by `bash bin/production/snapshot_session_state.sh`

Timestamp: 2026-06-01 08:20:03 CEST
Branch: grace-bc-parity-20260528

Git status:
 M README.md
 M bin/evaluation/compute_test_metrics_day.py
 M bin/evaluation/compute_test_metrics_day_fast.py
 M bin/evaluation/compute_test_metrics_day_rcm.py
 M bin/evaluation/compute_test_metrics_month.py
 M bin/evaluation/compute_test_metrics_month_rcm.py
 M bin/evaluation/compute_value_metrics.py
 M bin/evaluation/evaluate_futur_trend.py
 M bin/evaluation/generate_report.py
 M bin/evaluation/plot_exp5_historical_5curve.py
 M bin/evaluation/plot_exp5_pairwise_distribution_quantiles.py
 M bin/evaluation/plot_histograms.py
 M bin/preprocessing/bias_correction_ibicus.py
 M bin/preprocessing/build_dataset.py
 M bin/preprocessing/build_dataset_bc.py
 M bin/production/run_exp5_workflow.py
 M bin/production/run_exp5_workflow_grace.sh
 M bin/production/run_in_grace_env.sh
 M bin/production/submit_exp5_workflow_grace.sh
 M bin/production/submit_grace_venv_probe.sh
 M bin/training/predict_loop.py
 M doc/CALYPSO_RUNBOOK.md
 M doc/ENVIRONMENT_SETUP.md
 M doc/GRACE_TRAINING_ENGINEER_NOTE.md
 M doc/OBJECTIVE_PLAN.md
 M docs/egu26_short_course/ENVIRONMENT_SETUP.md
 M docs/egu26_short_course/LOCAL_WORKFLOW_RUNBOOK.md
 M docs/getting_started.rst
 M docs/training.rst
 M iriscc/datautils.py
 M iriscc/settings.py
?? EGU_Presentation_Slides.md
?? "IRISCC D4.1 Public Under EC Review.pdf"
?? SESSION_STATE.md
?? bin/evaluation/compare_archive_parity_outputs.py
?? bin/production/build_egu_cerfacs_review_pptx.py
?? bin/production/build_egu_html_preview.py
?? bin/production/create_globc_cpu_env.sh
?? bin/production/probe_grace_path_override.sh
?? bin/production/run_in_calypso_mixed_env.sh
?? bin/production/run_in_globc_cpu_env.sh
?? bin/production/snapshot_session_state.sh
?? bin/production/submit_exp5_5curve_grace.sh
?? bin/production/submit_exp5_pairwise_quantiles_grace.sh
?? bin/production/submit_grace_build_gpu_venv_probe.sh
?? bin/production/submit_grace_candidate_env_probe.sh
?? bin/production/submit_grace_clone_v22_gpu_probe.sh
?? bin/production/submit_grace_conda_activate_probe.sh
?? bin/production/submit_grace_gpu_env_probe.sh
?? bin/production/submit_grace_py311_sitepkg_probe.sh
?? bin/production/submit_grace_sitepkg_probe.sh
?? bin/production/submit_rcm_parity_grace.sh
?? data/
?? doc/FIGURE_VALIDATION_NOTES.md
?? doc/MORNING_BRIEF_2026-04-28.md
?? doc/PLOT_SELECTION_SHORTLIST.md
?? doc/PRESENTATION_FALLBACK_NOTES.md
?? doc/TRAINING_SLIDE_SUGGESTIONS.md

## Date-boundary audit

- The repo was audited again after the separate date-propagation PR landed on
  `master`.
- Core workflow/runtime code no longer hardcodes the `2014-12-31` /
  `2015-01-01` historical-scenario split.
- `run_exp5_workflow.py`, `run_exp5_perfect_model.py`, `predict_loop.py`,
  `iriscc/datautils.py`, and `iriscc/settings.py` now use configured windows
  from `settings.py`, with optional per-source scenario-start overrides for
  sources like `CORDEX` via:
  - `IDOWNSCALE_GCM_SCENARIO_START`
  - `IDOWNSCALE_CORDEX_SCENARIO_START`
  - `IDOWNSCALE_RCM_SCENARIO_START`
- Remaining `2014/2015` literals in the repo are mostly:
  - experiment configuration in `settings.py`
  - archive/path examples in docs
  - plotting/report scripts tied to current exp5 historical products
- EGU course notebook cells were checked; no stale command cells were found that
  relied on the old implicit date propagation behavior. The main EGU doc update
  was `docs/egu26_short_course/HELPER_SCRIPTS.md`.

## Master fix branch status

- A separate `master`-targeted worktree was created at:
  - `/scratch/globc/page/idownscale_master_boundary_fix`
- Branch:
  - `fix/date-boundary-configurable`
- Pushed commit:
  - `e4d2823` `Remove hardcoded date cutoffs from workflow paths`
- PR URL:
  - `https://github.com/cerfacs-globc/idownscale/pull/new/fix/date-boundary-configurable`

What that branch currently includes:

- explicit date-window examples in docs and runbooks
- workflow/runtime fixes so date-boundary logic no longer hardcodes
  `2014-12-31` / `2015-01-01`
- downstream scripts used by `run_exp5_full.sh` updated accordingly

Important caveat before merging:

- after pushing `e4d2823`, an additional local-only modification was started in
  `/scratch/globc/page/idownscale_master_boundary_fix/iriscc/settings.py`
  to widen the PR toward “no hardcoded model/reanalysis identity”
- that change is **not committed or pushed**
- current recommendation is to keep `fix/date-boundary-configurable` narrow and
  reviewable, and handle the broader “nothing hardcoded” generalization in a
  separate future PR

## Generalization principle

The intended long-term rule is:

- no hardcoded reanalysis name
- no hardcoded model name
- no hardcoded grid/projection identity
- no hardcoded historical/future cutoff dates
- no hardcoded filename conventions tied to a single source

However, this is broader than a small hotfix PR. Treat it as a dedicated
generalization follow-up rather than expanding the date-boundary fix indefinitely.

## Parallel work plan

The working strategy agreed at the end of the session is:

1. Keep the small `master` fix branch for date-boundary / explicit-args cleanup.
2. Prepare the broader “nothing hardcoded” generalization follow-up separately.
3. Continue ALADIN/RCM stabilization and perfect-model execution in parallel.

Items 1 and 2 should be handled while item 3 is still running.

## Perfect-model Kraken status

- The earlier perfect-model dataset built through ALADIN-BC files was archived
  as invalid because the coarse predictor repeated historical calendar-day
  fields in future periods.
- The corrected perfect-model workflow now builds:
  - `x`: native RCM degraded to the configurable coarse bridge grid, then
    remapped to the ML grid
  - `y`: native RCM remapped to the ML target grid as pseudo-truth
- Corrected dataset:
  - `/scratch/globc/page/idownscale_output/datasets/dataset_bc/dataset_perfect_model_rcm`
  - completed sample count: `44195 / 44195`
- Validation job:
  - `1884666` `pm_validate`
  - status: `COMPLETED`
  - output:
    - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/validation/perfect_model_samples_perfect_model_rcm_rcm_19800101_21001231.md`
  - inventory: `ok`
  - structure: `ok`
  - cross-period repeat check: `ok`
- Statistics job:
  - `1884667` `pm_stats`
  - status: `COMPLETED`
  - elapsed: `00:19:41`
  - peak RSS: about `1 GB`
  - output:
    - `/scratch/globc/page/idownscale_output/datasets/dataset_bc/dataset_perfect_model_rcm/statistics.json`

## Perfect-model model results

- UNet job:
  - training: `1884669` `pm_unet`, completed in `00:26:59`
  - prediction/comparison chains: `1884670`-`1884683`, all completed
  - aggregate table:
    - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_unet_perfect_model_rcm_rcm.md`
- MiniUNet job:
  - training: `1884684` `pm_miniunet`, completed in `00:26:59`
  - prediction/comparison chains: `1884685`-`1884698`, all completed
  - aggregate table:
    - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_miniunet_perfect_model_rcm_rcm.md`
- Combined table:
  - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.md`

Interpretation:

- UNet is the clean current candidate:
  - bias stays from `+0.044 K` to `+0.071 K` across historical and future
    windows
  - RMSE stays near `0.47-0.49 K`
  - degraded/coarse input RMSE is about `1.68-1.77 K`, so the ML gain is about
    `1.19-1.31 K`
- MiniUNet has good RMSE/correlation but a systematic cold bias:
  - bias from about `-0.239 K` to `-0.294 K`
  - this is above the working `0.10 K` bias tolerance
- Current queue state at this checkpoint:
  - `squeue -u page` is empty

Next recommended actions:

1. Treat corrected UNet perfect-model results as the first scientifically valid
   ML-vs-RCM benchmark.
2. Keep MiniUNet as an architecture comparison but not the bias-leading
   candidate unless it is calibrated or retrained.
3. Package the standalone perfect-model workflow as a reusable experiment that
   is not named after ALADIN, since ALADIN is only the current RCM source.
4. Continue the broader source/model/grid/date generalization in a separate PR
   from this perfect-model support.

## Active MiniSwinUNETR follow-up

- Fresh MiniSwinUNETR run on the corrected perfect-model dataset was launched
  after the UNet/MiniUNet validation checkpoint.
- First launch:
  - `1884742` `pm_mswin_train`
  - canceled after about 8 minutes because the job inherited repo-local output
    defaults and started writing under
    `/scratch/globc/page/idownscale_rerun/idownscale_output`
  - this was a path hygiene issue, not a scientific failure
- Clean relaunched training job:
  - `1884757` `pm_mswin_train2`
  - partition: `gpua30`
  - output root explicitly set to:
    - `/scratch/globc/page/idownscale_output`
  - dataset:
    - `/scratch/globc/page/idownscale_output/datasets/dataset_bc/dataset_perfect_model_rcm`
- Dependent prediction jobs:
  - `1884758` historical `20000101-20141231`
  - `1884760` future `20150101-20291231`
  - `1884762` future `20300101-20441231`
  - `1884764` future `20450101-20591231`
  - `1884766` future `20600101-20741231`
  - `1884768` future `20750101-20891231`
  - `1884770` future `20900101-21001231`
- Dependent comparison jobs:
  - `1884759`, `1884761`, `1884763`, `1884765`, `1884767`,
    `1884769`, `1884771`
  - partition: `prodshared`
- Result:
  - `1884757` was canceled after the epoch-6 gate
  - validation loss remained much too high:
    - epoch 0: `274.8`
    - epoch 3: `254.2`
    - epoch 6: `234.2`
  - by epoch 6, successful UNet/MiniUNet runs were already near `46` and `37`
    respectively
  - dependent prediction/comparison jobs were canceled because the model was
    not scientifically worth evaluating in future windows
  - peak RSS was about `6.1 GB`, so this was convergence/configuration, not a
    memory failure

## Perfect-model launcher fix

- `bin/production/run_exp5_perfect_model.py` now imports `CONFIG` correctly.
- The script docstring was generalized from ALADIN-specific wording to
  degraded-RCM/native-RCM perfect-model wording.
- Added `--sample-dir` so the standalone launcher can be pointed at a concrete
  dataset directory instead of depending implicitly on the active output-root
  environment variables.

## MONAI UNet follow-up

- A `monai_unet` architecture comparison was launched after MiniSwin was stopped.
- Training job:
  - `1884774` `pm_monai_train`
  - partition: `gpua30`
  - output root explicitly set to:
    - `/scratch/globc/page/idownscale_output`
- Dependent prediction jobs:
  - `1884775`, `1884777`, `1884779`, `1884781`, `1884783`, `1884785`,
    `1884787`
- Dependent comparison jobs:
  - `1884776`, `1884778`, `1884780`, `1884782`, `1884784`, `1884786`,
    `1884788`
- Apply the same early validation gate as MiniSwin:
  - keep if it bends toward the successful UNet/MiniUNet regime
  - cancel dependents if the curve remains clearly noncompetitive
- Result:
  - canceled at the epoch-6 gate
  - validation loss remained much too high for this setup:
    - epoch 0: `279.3`
    - epoch 3: `258.3`
    - epoch 6: `226.3`
  - this was treated as an underpowered/configuration issue, not a data or
    memory issue

## Active UNet robustness follow-up

- Two additional standard UNet stochastic replicates are running on the
  corrected `perfect_model_rcm` dataset.
- First replicate:
  - training job `1884803` `pm_unet2_train`
  - test name `unet_seed2_perfect_model_rcm`
  - running on `gpua30`
  - dependent predictions: `1884804`, `1884806`, `1884808`, `1884810`,
    `1884812`, `1884814`, `1884816`
  - dependent comparisons: `1884805`, `1884807`, `1884809`, `1884811`,
    `1884813`, `1884815`, `1884817`
- Second replicate:
  - training job `1884819` `pm_unet3_train`
  - test name `unet_rep3_perfect_model_rcm`
  - running on `gpua30`
  - dependent predictions: `1884820`, `1884822`, `1884824`, `1884826`,
    `1884828`, `1884830`, `1884832`
  - dependent comparisons: `1884821`, `1884823`, `1884825`, `1884827`,
    `1884829`, `1884831`, `1884833`
- Gate rule:
  - inspect `lightning_logs/version_best/metrics.csv` around epoch 6
  - keep the prediction/comparison chain only if the curve is close to the
    successful UNet regime
  - cancel dependent jobs if the curve remains clearly noncompetitive
- Current live read:
  - `unet_seed2_perfect_model_rcm` passed the epoch gate; epoch 7 validation
    loss reached `1.31`
  - `unet_rep3_perfect_model_rcm` also passed the epoch gate; epoch 6
    validation loss was about `40.08`
- Additional experiment:
  - training job `1884881` `pm_unet_on_train`
  - test name `unet_outputnorm_perfect_model_rcm`
  - standard UNet with `--output-norm`, launched to test whether target
    normalization improves bias/RMSE
  - dependent predictions: `1884882`, `1884884`, `1884886`, `1884888`,
    `1884890`, `1884892`, `1884894`
  - dependent comparisons: `1884883`, `1884885`, `1884887`, `1884889`,
    `1884891`, `1884893`, `1884895`
  - note: loss values are in normalized target units for this run, so do not
    compare its training loss numerically against non-output-normalized UNet
    losses
- Operational fix:
  - `train.py` performs a long post-fit test pass by default; for these
    perfect-model runs the meaningful evaluation is the separate
    prediction/comparison chain
  - seed2, rep3, and output-normalized runs were stopped after max-epoch
    checkpoint creation and their prediction/comparison chains were launched
    directly
  - future Slurm training submissions should pass `--skip-test` for this
    workflow
- Inference fix:
  - `predict_loop.py` now de-normalizes predictions when a checkpoint was
    trained with `output_norm=True`
  - before this fix, output-normalized predictions were written in normalized
    units and showed impossible `-283 K` biases
- Final aggregate table:
  - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.md`
  - standard UNet remains a strong valid benchmark:
    - bias `+0.044` to `+0.071 K`
    - RMSE about `0.467-0.488 K`
  - output-normalized UNet is currently the best candidate:
    - bias `-0.016` to `-0.005 K`
    - RMSE about `0.452-0.477 K`
    - RMSE gain over degraded input about `1.20-1.32 K`
  - rep3 is acceptable under the `0.10 K` bias rule:
    - bias `-0.050` to `-0.068 K`
    - RMSE about `0.514-0.532 K`
  - seed2 and MiniUNet are scientifically useful but biased cold:
    - seed2 bias about `-0.265` to `-0.320 K`
    - MiniUNet bias about `-0.239` to `-0.294 K`
- Figure checkpoint:
  - score dashboard:
    - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_method_comparison_perfect_model_rcm_rcm.png`
    - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_method_comparison_perfect_model_rcm_rcm.pdf`
  - distribution/PDF comparison:
    - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_distribution_pdf_perfect_model_rcm_rcm_tas.png`
    - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_distribution_pdf_perfect_model_rcm_rcm_tas.pdf`
  - PDF validation note:
    - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/validation/perfect_model_distribution_pdf_perfect_model_rcm_rcm_tas.md`
  - labels were cleaned up so the figures now say:
    - `RCM coarse-resolution input`
    - `RMSE reduction vs coarse-resolution input`
    - explicit RMSE and max-absolute-bias bars in the summary panel
  - legends were moved away from axis labels and dates, so both figures are
    now readable enough to keep as validation artifacts
- Metadata/provenance checkpoint:
  - `predict_loop.py` now supports `--var` and writes the prediction NetCDF
    using the selected variable name instead of a fully hardwired `tas` output
  - prediction datasets now receive lightweight provenance attrs, including:
    - experiment id
    - test name
    - simulation role
    - variable name
    - sample directory
    - perfect-model source-role config fields when available
  - this is the first step toward FAIRer outputs where diagnostics can read
    labels/units/provenance from metadata instead of relying on memory or shell
    command history
- Workflow integration checkpoint:
  - `run_exp5_perfect_model.py` now has explicit named steps for:
    - `validate_train_dataset`
    - `validate_eval_dataset`
    - `compare_predictions`
    - `aggregate_comparison`
    - `plot_score_comparison`
    - `plot_distribution`
  - the workflow train step now uses `--skip-test` and expects the Lightning
    `metrics.csv` log instead of a redundant post-fit test artifact
  - the workflow supports:
    - `--perfect-model-target-source`
    - validation start/end/historical-end overrides
    - validation unit labelling
  - `submit_exp5_perfect_model_kraken.sh` was added so this standalone
    perfect-model + validation chain can be launched as a Kraken Slurm job
- Generalization checkpoint:
  - perfect-model validation/comparison helpers no longer hardwire `x[1]` and
    `y[0]`; they resolve predictor/target channel indices from experiment
    metadata and the requested variable
  - aggregate comparison CSVs now carry generic columns (`ml_rmse`,
    `ml_bias`, `raw_rmse`, `raw_bias`, `rmse_reduction`) plus legacy `_K`
    aliases for backward compatibility
  - score plots now use a configurable bias tolerance and only draw the
    temperature-style `0.10 K` band when explicitly requested
  - comparison tables and markdown now carry variable label, unit, and selected
    provenance fields from the prediction NetCDFs
- Active validation exercise:
  - the integrated `build_eval_dataset -> validate_eval_dataset` chain is being
    exercised on Kraken against:
    - `/scratch/globc/page/idownscale_output/datasets/dataset_bc/dataset_perfect_model_rcm_test_rcm`
  - current runtime nuisance:
    - `xesmf` emits repeated `Input array is not C_CONTIGUOUS` warnings during
      sample packaging; these are performance warnings, not scientific errors

## Perfect-model BC recovery final status, 2026-06-05

- bounded Kraken recovery chain completed successfully
- final successful jobs:
  - prep / corrected BC rebuild: `1888048`
  - late-century ML recovery wave: `1888548` to `1888552`
  - dependent post-processing: `1888553`

### Bugs discovered and fixed

1. Wrong BC reference in perfect-model mode
   - degraded ALADIN was being corrected against `ERA5` and then compared
     against native ALADIN pseudo-truth
   - fixed with
     `CONFIG['perfect_model_rcm']['bc_reanalysis_source'] = 'rcm_aladin'`

2. Corrected perfect-model sample writer used the generic target path
   - caused
     `IndexError: 2-dimensional boolean indexing is not supported`
   - fixed in both `bias_correction_ibicus.py` and `bias_correction_sbck.py`
     by rebuilding `y` through the same native-model path as
     `build_dataset_pp.py`

3. Corrected sample packaging OOM
   - `1887337` failed after BC files were already written
   - fixed by loading orography once, packaging through a shared helper, and
     closing each one-day regridded dataset immediately

4. Corrected eval dataset contamination
   - `build_dataset_pp.py` incorrectly included `1980-1999` in eval datasets
     whenever `--corrected` was enabled
   - fixed so only explicit `--include-train-hist` adds `train_hist`

5. Future prediction sample-dir resolution bug
   - `predict_loop.py` defaulted future perfect-model runs to a historical
     sample directory
   - fixed with date-aware sample-dir resolution

6. Downstream future diagnostics had the same assumption
   - `plot_perfect_model_distribution_pdf.py` and
     `compare_perfect_model_climate_signal.py` now resolve sample directories
     per window

7. Recovery relaunch initially assumed too many future windows
   - observed corrected future dataset for this rerun path only exposed
     `2090-2100`
   - successful final recovery wave was therefore narrowed to
     `20900101_21001231`

### Final scientific status

- final combined windows currently available:
  - `20000101_20141231`
  - `20900101_21001231`
- late-century raw coarse input and BC baseline are identical in the table:
  - bias `0.230943 K`
  - RMSE `1.760735 K`
- late-century best ML result remains `UNet + output norm`:
  - bias `-0.037046 K`
  - RMSE `0.452707 K`
- climate-signal truth mean warming:
  - `3.619777 K`
- climate-signal RMSE:
  - raw input: `0.160099 K`
  - BC baseline: `0.160117 K`
  - `UNet + output norm`: `0.079900 K`
  - `MiniUNet`: `0.075264 K`

### Final technical status

- no active jobs remain
- final outputs present:
  - combined comparison table:
    - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.csv`
  - score plot:
    - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_method_comparison_perfect_model_rcm_rcm.png`
  - PDF plot:
    - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_distribution_pdf_perfect_model_rcm_rcm_tas.png`
  - climate-signal table:
    - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_climate_signal_perfect_model_rcm_rcm_20000101_20141231_vs_20900101_21001231.csv`
<!-- SESSION_STATE_AUTOGEN_END -->
