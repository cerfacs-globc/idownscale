# Release Notes - Stabilization and Research Readiness (v1.1.0)

## Summary
This release focuses on restoring scientific integrity to Experiment 5, implementing a robust automated validation framework, and ensuring codebase portability and cleanliness for high-impact climate research.

## Critical Scientific Fixes
- **Temperature Bias Regression**: Resolved a catastrophic -180K bias in EXP5. Current validation shows a mean bias of **0.12 K** and Spatial RMSE of **0.74 K**.
- **Loss Function Logic**: Reverted to `masked_mse` as the stable baseline for EXP5.
- **Normalization Stability**: Standardized `fill_value` to `0.0` and corrected dataset sample paths in `hparams.py`.

## New Features & Infrastructure
- **Automated Pipeline Integrity Suite**: New utility `bin/utils/check_pipeline_integrity.py` performs deep health checks (data range, artifact existence, convergence) at every phase.
- **Structured Logging**: All `stdout`/`stderr` logs are now automatically routed to `logs/$EXP/$RUN_ID/` to prevent repository pollution.
- **Consolidated Validation Output**: All validation plots, CSV metrics, and PDF reports are now centralized in `output/$EXP/validation/`.
- **Workspace Portability**: New `bin/utils/setup_workspace.sh` utility allows collaborators to adapt hardcoded paths to their local environments instantly.

## Quality & Testing
- **Automated Unit Testing**: Enhanced `tests/` suite with 100% coverage on critical scientific parameters (`settings.py`, `hparams.py`).
- **Code Standards**: 100% Ruff/Lint compliance on all core stabilization modules.
- **Repository Cleanup**: Removed redundant `doc/` directories and optimized `.gitignore`.

## Deployment Instructions
1. Run `./bin/utils/setup_workspace.sh` if deploying in a new environment.
2. Execute `./run_exp5_full.sh` to run the full pipeline with automated integrity checks.
3. Review final results in `output/exp5/validation/`.
