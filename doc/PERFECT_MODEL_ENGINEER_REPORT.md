# Perfect-model experiment report

Date: 2026-06-08
Platform: `kraken`  
Experiment: `perfect_model_rcm`

## 1. Objective

This experiment is our current perfect-model benchmark for downscaling. The principle is to use an RCM as its own controlled reference: we degrade the native RCM field to the coarse bridge grid used by the ML models, then compare the downscaled products back to the native-resolution RCM pseudo-truth.

This setup is useful because we know the high-resolution future answer. It lets us compare raw coarse input, bias correction, and ML downscaling in a future climate, including the climate-change signal.

In the current run, the reference RCM is ALADIN and the variable is near-surface air temperature (`tas`, K). The workflow itself should not rely on ALADIN-specific assumptions; ALADIN is the current source, not the concept.

## 2. Implementation Status

The standalone runner is `bin/production/run_exp5_perfect_model.py`. The current workflow can build BC datasets, apply BC, build perfect-model training/evaluation samples, validate samples, compute statistics, train, predict, and run dedicated perfect-model diagnostics.

The main diagnostic scripts are `validate_perfect_model_samples.py`, `compare_perfect_model_predictions_vs_truth.py`, `aggregate_perfect_model_comparisons.py`, `plot_perfect_model_comparison.py`, `plot_perfect_model_distribution_pdf.py`, `compare_perfect_model_climate_signal.py`, and `compare_perfect_model_window_statistics.py`.

The comparison now includes raw coarse-resolution input, the default CDFt BC baseline, an SBCK CDFt baseline, and five ML runs: `unet_outputnorm_perfect_model_rcm`, `unet_perfect_model_rcm`, `unet_rep3_perfect_model_rcm`, `miniunet_perfect_model_rcm`, and `unet_seed2_perfect_model_rcm`.

On Kraken, the interpreter used for this work is `/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1/bin/python`. The output root used for the validated artifacts is `/scratch/globc/page/idownscale_output`.

## 3. How To Run

A typical direct run is:

```bash
/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1/bin/python \
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

For reproducible runs, set the output root explicitly:

```bash
export IDOWNSCALE_OUTPUT_DIR=/scratch/globc/page/idownscale_output
```

## 4. Validated Outputs

The main comparison table is `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.csv`.

The score plot is `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_method_comparison_perfect_model_rcm_rcm.png`.

The PDF/distribution plot is `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_distribution_pdf_perfect_model_rcm_rcm_tas.png`.

The climate-signal diagnostic is `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_climate_signal_perfect_model_rcm_rcm_19810101_20101231_vs_20800101_21001231.csv`.

The all-window variability diagnostic is `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_window_statistics_perfect_model_rcm_rcm.csv`.

The combined prediction-vs-truth table currently covers `20000101_20141231` and `20900101_21001231`. The climate-signal diagnostic uses `19810101_20101231` as the reference and `20800101_21001231` as the future period. The window-statistics diagnostic covers all eight windows from `19810101_20101231` to `20900101_21001231`.

## 5. Main Results

For `2000-2014`, the raw coarse-resolution input has a bias of `0.338547 K` and an RMSE of `1.773409 K`. The default CDFt baseline has a bias of `0.258485 K` and an RMSE of `1.855405 K`. The SBCK CDFt baseline is essentially identical, with a bias of `0.258475 K` and an RMSE of `1.855406 K`.

The best ML run in that historical window is `UNet + output norm`, with a bias of `-0.024973 K` and an RMSE of `0.408541 K`. The `UNet cold replicate` is also strong, with an RMSE of `0.451416 K`. `MiniUNet` remains competitive in RMSE (`0.477624 K`) but has a warmer mean bias (`0.139126 K`).

For `2090-2100`, the raw coarse-resolution input has a bias of `0.301807 K` and an RMSE of `1.681219 K`. The default CDFt baseline has a bias of `0.230943 K` and an RMSE of `1.760735 K`. SBCK CDFt has a smaller bias (`0.201856 K`) but a slightly larger RMSE (`1.775895 K`).

The best ML run in late century is again `UNet + output norm`, with a bias of `-0.037046 K` and an RMSE of `0.452707 K`. `UNet cold replicate` follows closely (`0.479083 K` RMSE), and `MiniUNet` remains useful but warmer (`0.148576 K` bias, `0.506672 K` RMSE).

## 6. Climate Signal

The climate-signal comparison is `2080-2100` relative to `1981-2010`. The pseudo-truth warming is `4.007120 K`.

The raw coarse input has a signal bias of `-0.046903 K`, a signal RMSE of `0.143515 K`, and a spatial correlation of `0.965594`. The default BC baseline preserves the broad warming but does not improve the signal RMSE here (`0.159253 K`). SBCK CDFt is slightly weaker in this diagnostic, with a signal RMSE of `0.185034 K`.

The ML methods improve the spatial climate-change signal. `UNet + output norm` has a signal RMSE of `0.113563 K`, and `MiniUNet` is the best on this specific signal metric with `0.100773 K`. Both have correlations close to `0.988`.

## 7. Variability Diagnostics

The all-window statistics table includes mean bias, RMSE, mean absolute error, day-to-day variability bias, spatial variability bias, and annual variability bias.

The raw and BC baselines retain a strong negative spatial-variability bias across windows. The ML methods reduce that bias substantially. `UNet + output norm` is the most balanced method across mean error, RMSE, climate signal, and variability behavior.

IBICUS and SBCK are nearly identical in the historical and early future windows. SBCK becomes slightly worse than the default CDFt path in the later future windows, especially for RMSE and spatial-variability bias.

## 8. Scientific Interpretation

The perfect-model workflow is now operational and scientifically usable for this controlled RCM benchmark. The main result is robust: ML downscaling clearly improves over the degraded coarse-resolution RCM input and over the BC baselines tested here.

The output-normalized UNet remains the best all-around candidate. It combines low RMSE, small mean bias, good climate-signal fidelity, and good variability behavior. MiniUNet is interesting because it performs very well on the climate-signal RMSE, but its mean bias is less attractive for production use.

The two CDFt implementations are close enough that SBCK does not change the conclusion. In this setup, BC alone is not competitive with ML, and BC does not improve the perfect-model climate signal relative to the raw coarse input.

## 9. Next Steps

The same protocol should now be reused for other variables, additional ML methods, and other RCM references. The workflow is close to that target: BC is selectable at workflow level, and evaluation can compare several BC baselines side by side.
