# Perfect-model implementation notes

Date: 2026-06-08

This note keeps the implementation history separate from the science-facing perfect-model report. It is meant to help us remember why the workflow is shaped the way it is, without putting debugging history in the main result note.

## 1. Main Corrections

The first important correction was the BC reference used in perfect-model mode. The degraded ALADIN predictor was initially being bias-corrected against ERA5 while the evaluation was done against native ALADIN. That was not a valid perfect-model baseline. The configuration now uses `rcm_aladin` as the BC reference for `perfect_model_rcm`.

The corrected perfect-model sample writer also needed to rebuild `y` through the native-model path. The generic observation target loader was not appropriate for this case and could fail on the curvilinear-grid masking path. The corrected sample writers in `bias_correction_ibicus.py` and `bias_correction_sbck.py` now follow the same pseudo-truth logic as `build_dataset_pp.py`.

Corrected sample packaging was made lighter after an out-of-memory failure. The packaging code now avoids repeated orography loading and closes temporary one-day regridded datasets promptly.

The corrected eval dataset had also been contaminated by `1980-1999` samples because `--corrected` implicitly added `train_hist`. That is no longer the case. The train historical block is only included when `--include-train-hist` is explicitly requested.

Future prediction and diagnostics now resolve sample directories by date window. This fixed hidden assumptions in `predict_loop.py`, `plot_perfect_model_distribution_pdf.py`, and `compare_perfect_model_climate_signal.py`.

## 2. Multi-BC Support

The workflow rule is still simple: one production run uses one selected BC method. The evaluation layer is more flexible: it can compare multiple BC baselines generated from separate runs.

Tagged BC outputs were added to the IBICUS and SBCK correction scripts. The comparison script can write tagged BC rows, and the aggregate script creates synthetic BC rows such as `bc_baseline` and `bc_baseline_sbck_cdft`.

One aggregation bug was found after SBCK was generated. The SBCK chunks were written correctly, but their filenames ended with suffixes such as `_bc_sbck_cdft.csv`, and the aggregator only looked for files ending exactly at the window token. The glob pattern was broadened so tagged BC chunks are included. The combined table now contains both BC baselines.

## 3. Final Diagnostics

The final diagnostic set now includes the combined prediction-vs-truth table, the climate-signal comparison, and the all-window statistics table. On 2026-06-08, the final files were verified with SBCK included in all three diagnostics.

The window-statistics table is especially useful because it adds day-to-day variability, spatial variability, and annual variability diagnostics to the usual RMSE and bias comparison.

## 4. Runtime Notes From Kraken

Kraken showed intermittent filesystem symptoms during the last refreshes. Some Python processes entered `D` state while streaming many sample files from scratch, and several tiny Slurm reruns failed immediately with `RaisedSignal:53(Real-time_signal_19)`. The outputs eventually landed, but this is a reminder that the filesystem behavior was part of the runtime risk.

A separate Git metadata issue also appeared on the scratch-hosted checkout. The local `.git` directory had grown very large because stale `tmp_pack_*` files were left behind after an interrupted Git pack operation. The repository was repaired by pushing from a clean clone and replacing the damaged local `.git` metadata.

The safer long-term layout is now documented and partly implemented: keep the Git repository in backed-up `home`, and keep raw data, outputs, weights, runs, predictions, metrics, and temporary files on scratch.

## 5. Remaining Engineering Work

The current perfect-model result is complete for the present benchmark. The next engineering improvement would be to make the all-window future sample materialization a first-class workflow step instead of relying on recovery scripts and manually prepared validation windows.

The same workflow should then be tested on another variable. That will be the real proof that the validation and plotting layers are no longer temperature-specific.
