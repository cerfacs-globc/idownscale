# Restart Handoff - 2026-06-08

This note is meant to restart a new Codex conversation without relying on the
previous chat context.

## Repository State

- Active repository: `/home/globc/page/idownscale`
- Active branch after cleanup PR merge: `master`
- Latest merged cleanup PR: `#10`, "Normalize runtime layout and public docs"
- Cleanup commit on the PR branch: `86b27c9 Normalize runtime layout and docs`
- `master` has been fast-forwarded after the merge.

The previous scratch checkout at `/scratch/globc/page/idownscale_rerun` is
deprecated. Do not use it as the active code checkout. It remains useful only as
preserved work material until all remaining scratch artifacts are confirmed
copied or no longer needed.

## Current Layout

The intended split is now:

```text
/home/globc/page/idownscale/                  # Git checkout, backed up home
/scratch/globc/page/idownscale_runtime/       # Active runtime tree
├── rawdata/                                  # Input data root, if using runtime-local raw data
├── output/                                   # Datasets, predictions, metrics, runs, weights
└── graphs/                                   # Figures and graphical diagnostics

/scratch/globc/page/idownscale_rawdata/       # Shared/raw input data root used locally
/scratch/globc/page/idownscale_work_materials # Preserved non-committed working material
```

Local ignored settings are in:

```text
/home/globc/page/idownscale/iriscc/settings_local.py
```

The local settings point to:

```bash
IDOWNSCALE_RUNTIME_ROOT=/scratch/globc/page/idownscale_runtime
IDOWNSCALE_OUTPUT_DIR=/scratch/globc/page/idownscale_runtime/output
IDOWNSCALE_GRAPHS_DIR=/scratch/globc/page/idownscale_runtime/graphs
IDOWNSCALE_REGRID_WEIGHTS_DIR=/scratch/globc/page/idownscale_runtime/output/regrid_weights
```

The public example is:

```text
iriscc/settings_local.py.example
```

## Runtime And Scratch Cleanup Status

The old `/scratch/globc/page/idownscale_output` was trimmed to explicit legacy
material only:

- `_legacy_flat_backup_20260602`
- `_legacy_root_artifacts_20260608`
- `dataset_exp5_30y_arch`

Active runtime artifacts were moved under:

```text
/scratch/globc/page/idownscale_runtime/output
/scratch/globc/page/idownscale_runtime/graphs
```

The deprecated `/scratch/globc/page/idownscale_rerun` still had many old dirty
tracked and untracked files. Useful material was preserved in:

```text
/scratch/globc/page/idownscale_work_materials/rerun_20260608
```

Do not delete `/scratch/globc/page/idownscale_rerun` until any remaining useful
material in its `scratch/` subdirectory has been inspected and either copied or
explicitly discarded.

## Environment

Primary Kraken Python environment used during the perfect-model work:

```text
/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1
```

Activation can be done with:

```bash
conda activate /scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1
```

or by using the interpreter directly:

```bash
/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1/bin/python
```

The merged wrappers no longer default to personal environment paths. Pass
`IDOWNSCALE_VENV_PATH` or `PYTHON_BIN` explicitly when launching site-specific
Slurm jobs.

## Perfect-Model Status

The current perfect-model reference documentation is:

```text
doc/PERFECT_MODEL_ENGINEER_REPORT.md
doc/PERFECT_MODEL_IMPLEMENTATION_NOTES.md
```

The perfect-model benchmark now compares:

- raw degraded RCM input
- CDFt BC baseline through the default/IBICUS path
- CDFt BC baseline through SBCK
- ML methods including UNet variants, MiniUNet, and CDDPM

The current scientific conclusion is that ML downscaling clearly improves over
raw degraded RCM input and over the BC-only baselines in the controlled
perfect-model ALADIN/RCM benchmark. The output-normalized UNet is the best
overall candidate in the corrected results. MiniUNet is strong on the climate
signal and remains very competitive on direct field RMSE. The corrected CDDPM
workflow is scientifically acceptable as a comparison method, but it still
underperforms the best UNet variants on direct field error.

Validated outputs should now live under:

```text
$IDOWNSCALE_OUTPUT_DIR/metrics/perfect_model_rcm/
$IDOWNSCALE_GRAPHS_DIR/metrics/perfect_model_rcm/
```

Important validated files from the corrected June 11 rerun are:

- combined comparison:
  - `$IDOWNSCALE_OUTPUT_DIR/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.csv`
- climate signal:
  - `$IDOWNSCALE_OUTPUT_DIR/metrics/perfect_model_rcm/comparison_tables/perfect_model_climate_signal_perfect_model_rcm_rcm_19810101_20101231_vs_20800101_21001231.csv`
- all-window variability:
  - `$IDOWNSCALE_OUTPUT_DIR/metrics/perfect_model_rcm/comparison_tables/perfect_model_window_statistics_perfect_model_rcm_rcm.csv`

Key plots from the corrected rerun are:

- `$IDOWNSCALE_GRAPHS_DIR/metrics/perfect_model_rcm/perfect_model_method_comparison_perfect_model_rcm_rcm.png`
- `$IDOWNSCALE_GRAPHS_DIR/metrics/perfect_model_rcm/perfect_model_distribution_pdf_perfect_model_rcm_rcm_tas.png`
- `$IDOWNSCALE_GRAPHS_DIR/metrics/perfect_model_rcm/perfect_model_climate_signal_perfect_model_rcm_rcm_19810101_20101231_vs_20800101_21001231.png`

The all-window sample build used the corrected BC-conditioned dataset:

- `$IDOWNSCALE_OUTPUT_DIR/datasets/dataset_bc/dataset_perfect_model_rcm_all_windows_bcml_audit`

Prediction, training, dataset-build, statistics, and workflow stages now all
emit provenance files and resolved-context stdout blocks. The main places to
look are:

- training:
  - `$IDOWNSCALE_OUTPUT_DIR/runs/perfect_model_rcm/<test_name>/provenance_train.prov.json`
- prediction:
  - sidecar next to each prediction NetCDF, ending in `.prov.json`
- dataset build:
  - `.../dataset_perfect_model_rcm_*/provenance_build_dataset.prov.json`
- workflow:
  - `$IDOWNSCALE_OUTPUT_DIR/metrics/perfect_model_rcm/comparison_tables/workflow_<test_name>.prov.json`

## Merged Cleanup Work

PR `#10` did the following:

- changed default output layout to `$IDOWNSCALE_RUNTIME_ROOT/output`
- changed graph layout to `$IDOWNSCALE_RUNTIME_ROOT/graphs`
- changed regrid weights to `output/regrid_weights`
- removed broken README image references
- made README and public docs more platform-neutral
- updated EGU short-course docs and Mercure deployment plan
- removed hard-coded personal archive defaults from BC validation
- made Grace/Kraken wrappers avoid personal venv/path defaults
- kept `doc/CALYPSO_RUNBOOK.md` and the two Grace notes as historical/operator
  traceability documents

Validation before PR merge:

- `git diff --check`
- `bash -n run_exp5_full.sh setup_env.sh bin/production/*.sh`
- `/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1/bin/python -m compileall -q iriscc bin`
- settings/model import smoke test
- local markdown relative-link scan
- GitHub CI passed after merge

## Current Workflow Notes

The corrected standalone perfect-model workflow now has one important date
override mechanism:

- `bin/production/run_exp5_perfect_model.py` accepts
  - `--work-startdate`
  - `--work-enddate`

These are used by the standalone `predict`, `compare_predictions`, and metrics
steps so future-window launches do not silently fall back to the historical
`20000101-20141231` window.

The Kraken submit wrapper forwards the same values through:

- `WORK_STARTDATE`
- `WORK_ENDDATE`

The recent model-file cleanup only touched `if __name__ == "__main__"` demo
blocks to remove hard-coded personal sample paths. It did not change model class
behavior.

## Important Cautions

- Do not use `/scratch/globc/page/idownscale_rerun` as the active checkout.
- Do not reintroduce personal `/scratch/globc/page/...`, `/gpfs-calypso/...`, or
  `/scratch/globc/garcia/...` paths as code defaults.
- `settings.py` and ignored `settings_local.py` are the intended places for
  local path configuration.
- Historical notes can mention old machine-specific paths when needed for
  traceability, but runnable public defaults should stay portable.
- Before launching long Slurm jobs, verify `IDOWNSCALE_OUTPUT_DIR`,
  `IDOWNSCALE_GRAPHS_DIR`, `IDOWNSCALE_RAW_DIR`, and the Python environment.
