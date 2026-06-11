# Perfect-model experiment report

Date: 2026-06-11
Platform: `kraken`  
Experiment: `perfect_model_rcm`

## 1. Objective

This experiment is our current perfect-model benchmark for downscaling. The principle is to use an RCM as its own controlled reference: we degrade the native RCM field to the coarse bridge grid used by the ML models, then compare the downscaled products back to the native-resolution RCM pseudo-truth.

This setup is useful because we know the high-resolution future answer. It lets us compare raw coarse input, bias correction, and ML downscaling in a future climate, including the climate-change signal.

In the current run, the reference RCM is ALADIN and the variable is near-surface air temperature (`tas`, K). The workflow itself should not rely on ALADIN-specific assumptions; ALADIN is the current source, not the concept.

## 2. Implementation Status

The standalone runner is `bin/production/run_exp5_perfect_model.py`. The current workflow can build BC datasets, apply BC, build perfect-model training/evaluation samples, validate samples, compute statistics, train, predict, and run dedicated perfect-model diagnostics.

The main diagnostic scripts are `validate_perfect_model_samples.py`, `compare_perfect_model_predictions_vs_truth.py`, `aggregate_perfect_model_comparisons.py`, `plot_perfect_model_comparison.py`, `plot_perfect_model_distribution_pdf.py`, `compare_perfect_model_climate_signal.py`, and `compare_perfect_model_window_statistics.py`.

The comparison now includes raw coarse-resolution input, the default CDFt BC baseline, an SBCK CDFt baseline, and six ML runs: `unet_outputnorm_perfect_model_rcm`, `unet_perfect_model_rcm`, `unet_rep3_perfect_model_rcm`, `miniunet_perfect_model_rcm`, `unet_seed2_perfect_model_rcm`, and `cddpm_perfect_model_rcm`.

The replicate names need one explicit note because they are easy to misread later:

- `unet_rep3_perfect_model_rcm` is the third UNet replicate, using the same architecture and training protocol as the base UNet but a different random initialization and training seed.
- `unet_seed2_perfect_model_rcm` is the UNet replicate trained with seed 2. In the engineering notes it is sometimes called the "cold replicate" because it tended to produce a slightly cooler mean state than the base run.

On Kraken, the interpreter used for this work was the local GPU environment installed under the project scratch area. The validated artifacts are written under the runtime split layout: `$IDOWNSCALE_OUTPUT_DIR` for datasets, predictions and tables, and `$IDOWNSCALE_GRAPHS_DIR` for figures.

The scientifically corrected perfect-model path is now BC+ML for all ML methods, including CDDPM. The packaged predictor tensor is three-channel: elevation, degraded coarse-temperature input, and BC temperature input. The target is the native-resolution pseudo-truth from the same RCM.

## 3. How To Run

A typical direct run is:

```bash
python \
  bin/production/run_exp5_perfect_model.py \
  --exp perfect_model_rcm \
  --var tas \
  --ssp ssp585 \
  --simu rcm \
  --test-name unet_perfect_model_rcm \
  --train-model unet \
  --train-max-epoch 30 \
  --train-batch-size 32 \
  --train-learning-rate 0.0008 \
  --train-output-norm
```

The Kraken submitter is:

```bash
sbatch bin/production/submit_exp5_perfect_model_kraken.sh
```

For reproducible runs, set the runtime roots explicitly:

```bash
export IDOWNSCALE_RUNTIME_ROOT=/path/to/idownscale_runtime
export IDOWNSCALE_OUTPUT_DIR=$IDOWNSCALE_RUNTIME_ROOT/output
export IDOWNSCALE_GRAPHS_DIR=$IDOWNSCALE_RUNTIME_ROOT/graphs
```

## 4. Validated Outputs

The main comparison table is `$IDOWNSCALE_OUTPUT_DIR/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.csv`.

The score plot is `$IDOWNSCALE_GRAPHS_DIR/metrics/perfect_model_rcm/perfect_model_method_comparison_perfect_model_rcm_rcm.png`.

The PDF/distribution plot is `$IDOWNSCALE_GRAPHS_DIR/metrics/perfect_model_rcm/perfect_model_distribution_pdf_perfect_model_rcm_rcm_tas.png`.

The climate-signal diagnostic is `$IDOWNSCALE_OUTPUT_DIR/metrics/perfect_model_rcm/comparison_tables/perfect_model_climate_signal_perfect_model_rcm_rcm_19810101_20101231_vs_20800101_21001231.csv`.

The all-window variability diagnostic is `$IDOWNSCALE_OUTPUT_DIR/metrics/perfect_model_rcm/comparison_tables/perfect_model_window_statistics_perfect_model_rcm_rcm.csv`.

The combined prediction-vs-truth table currently covers `20000101_20141231` and `20900101_21001231`. The climate-signal diagnostic uses `19810101_20101231` as the reference and `20800101_21001231` as the future period. The window-statistics diagnostic covers all eight windows from `19810101_20101231` to `20900101_21001231`.

## 5. Provenance

Every important workflow stage now emits two provenance traces:

1. A resolved-context block in stdout delimited by:
   - `=== IDOWNSCALE RESOLVED CONTEXT START ===`
   - `=== IDOWNSCALE RESOLVED CONTEXT END ===`
2. A `.prov.json` sidecar or workflow provenance file written on disk, with its path echoed as:
   - `provenance_provjson=...`

What is written today:

- dataset build:
  - `$IDOWNSCALE_OUTPUT_DIR/datasets/dataset_bc/.../provenance_build_dataset.prov.json`
- statistics:
  - `$IDOWNSCALE_OUTPUT_DIR/datasets/dataset_bc/.../provenance_statistics.prov.json`
- training:
  - `$IDOWNSCALE_OUTPUT_DIR/runs/perfect_model_rcm/<test_name>/provenance_train.prov.json`
- prediction:
  - sidecar next to each NetCDF prediction file, for example
    `$IDOWNSCALE_OUTPUT_DIR/prediction/..._perfect_model_rcm_<test_name>_rcm.prov.json`
- workflow driver:
  - `$IDOWNSCALE_OUTPUT_DIR/metrics/perfect_model_rcm/comparison_tables/workflow_<test_name>.prov.json`

How to interpret the provenance:

- `parameters` are the explicit CLI arguments that were passed to the script.
- `settings` are the resolved runtime choices after defaults and config expansion. This is the first place to check when a run appears to have used the wrong dates, sample directory, checkpoint, statistics directory, batch size, or diffusion settings.
- `inputs` and `outputs` point to the concrete files or directories the step read and wrote.
- `runtime` captures the host, user, output root, raw-data root, and Slurm job identifiers when present.

For debugging, the most important fields are usually:

- prediction:
  - `startdate`, `enddate`, `sample_dir`, `checkpoint_dir`, `statistics_json`, `diffusion_num_samples`, `output_range`
- training:
  - `sample_dir`, `statistics_json`, `runs_dir`, `model`, `seed`, `n_steps`
- dataset build:
  - `output_dir`, `perfect_model_target_source`, `orog_file`, `target_file`

The provenance layer was added because earlier failures were caused by silent defaults. The goal is that a future user can reconstruct exactly which window, data source, checkpoint, and normalization statistics were used without relying on memory or chat history.

## 6. Main Results

For `2000-2014`, the raw coarse-resolution input has a bias of `0.338547 K` and an RMSE of `1.773409 K`. The default CDFt BC baseline has a bias of `0.258485 K` and an RMSE of `1.855405 K`. The SBCK CDFt baseline is essentially identical, with a bias of `0.258475 K` and an RMSE of `1.855406 K`.

The best ML run in that historical window is `UNet + output norm`, with a bias of `-0.088262 K` and an RMSE of `0.434743 K`. `MiniUNet` follows very closely (`-0.026151 K` bias, `0.440002 K` RMSE), and the seed-2 UNet replicate is also strong (`0.028658 K` bias, `0.451776 K` RMSE). The corrected CDDPM run is no longer catastrophically biased, but it remains clearly weaker than the UNet family on direct field error (`-0.211169 K` bias, `0.910677 K` RMSE).

For `2090-2100`, the raw coarse-resolution input has a bias of `0.301807 K` and an RMSE of `1.681219 K`. The default CDFt BC baseline has a bias of `0.230943 K` and an RMSE of `1.760735 K`. SBCK CDFt has a smaller bias (`0.201856 K`) but a slightly larger RMSE (`1.775895 K`).

The best ML run in late century is again `UNet + output norm`, with a bias of `-0.093192 K` and an RMSE of `0.463097 K`. `MiniUNet` is almost tied (`-0.048015 K` bias, `0.463998 K` RMSE), and the seed-2 replicate remains close (`0.007686 K` bias, `0.470001 K` RMSE). CDDPM remains behind the deterministic UNets but is still much better than BC-only (`-0.211870 K` bias, `0.920130 K` RMSE).

## 7. Climate Signal

The climate-signal comparison is `2080-2100` relative to `1981-2010`. The pseudo-truth warming is `4.007120 K`.

The raw coarse input has a signal bias of `-0.046903 K`, a signal RMSE of `0.143515 K`, and a spatial correlation of `0.965594`. The default BC baseline preserves the broad warming but does not improve the signal RMSE here (`0.159253 K`). SBCK CDFt is slightly weaker in this diagnostic, with a signal RMSE of `0.185034 K`.

The ML methods improve the spatial climate-change signal. `MiniUNet` is the best on this specific signal metric with a signal RMSE of `0.100267 K`. `UNet + output norm` follows closely at `0.104324 K`. The corrected CDDPM signal RMSE is `0.125902 K`, which is better than raw coarse input and better than both BC-only baselines, even though its direct field RMSE is still weaker than the best UNets.

## 8. Variability Diagnostics

The all-window statistics table includes mean bias, RMSE, mean absolute error, day-to-day variability bias, spatial variability bias, and annual variability bias.

The raw and BC baselines retain a strong negative spatial-variability bias across windows. The ML methods reduce that bias substantially. In the late-century `2090-2100` window, `UNet + output norm` has the smallest RMSE (`0.452707 K`) and a nearly neutral spatial-variability bias (`0.004180`). The seed-2 UNet replicate is also good but suppresses spatial variability a bit more strongly (`-0.060602`). CDDPM shows a stronger residual field error (`0.920130 K` RMSE) but still improves strongly over raw and BC baselines.

IBICUS and SBCK are nearly identical in the historical and early future windows. SBCK becomes slightly worse than the default CDFt path in the later future windows, especially for RMSE and spatial-variability bias.

## 9. Scientific Interpretation

The perfect-model workflow is now operational and scientifically usable for this controlled RCM benchmark. The main result is robust: BC+ML clearly improves over the degraded coarse-resolution RCM input and over the BC-only baselines tested here.

The output-normalized UNet remains the best all-around candidate. It combines low RMSE, small mean bias, good climate-signal fidelity, and good variability behavior. MiniUNet is interesting because it performs extremely well on the climate-signal RMSE and stays competitive on direct field RMSE.

The two CDFt implementations are close enough that SBCK does not change the conclusion. In this setup, BC alone is not competitive with ML, and BC does not improve the perfect-model climate signal relative to the best ML methods.

The corrected CDDPM result changes the earlier interpretation substantially. The previous very large warm bias was caused by workflow and implementation errors. With the corrected BC-conditioned workflow, model-specific statistics, and corrected inference path, CDDPM now behaves reasonably. It is still not the leading model in this benchmark, but it is scientifically acceptable as a comparison method and it does improve over raw and BC-only baselines.

## 10. Next Steps

The same protocol should now be reused for other variables, additional ML methods, and other RCM references. The workflow is close to that target: BC is selectable at workflow level, and evaluation can compare several BC baselines side by side.
