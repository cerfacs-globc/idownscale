# Perfect-Model Experiment Report

Date: 2026-06-06  
Platform: `kraken`  
Experiment: `perfect_model_rcm`

## 1. Purpose

This experiment is our perfect-model benchmark for downscaling.

The idea is simple:

- take one RCM available in historical and future conditions
- degrade it to the coarse bridge grid used as ML input
- keep the native-resolution RCM as pseudo-truth
- compare:
  - raw coarse-resolution RCM input
  - BC baseline(s)
  - ML downscaling

This is the cleanest way to ask:

- does ML improve over the coarse RCM input?
- does ML improve over BC alone?
- does ML preserve the climate-change signal?

At the moment the reference RCM is `ALADIN`, but the workflow is written so
that the logic is not tied to ALADIN itself.

## 2. Technical context

### Workflow

Main runner:

- `bin/production/run_exp5_perfect_model.py`

Main diagnostics:

- `bin/evaluation/validate_perfect_model_samples.py`
- `bin/evaluation/compare_perfect_model_predictions_vs_truth.py`
- `bin/evaluation/aggregate_perfect_model_comparisons.py`
- `bin/evaluation/plot_perfect_model_comparison.py`
- `bin/evaluation/plot_perfect_model_distribution_pdf.py`
- `bin/evaluation/compare_perfect_model_climate_signal.py`
- `bin/evaluation/compare_perfect_model_window_statistics.py`

### Current methods compared

Raw / BC baselines:

- raw coarse-resolution RCM input
- `bc_baseline` = current default CDFt path
- `bc_baseline_sbck_cdft` = supplemental SBCK CDFt comparison

ML methods:

- `unet_outputnorm_perfect_model_rcm`
- `unet_perfect_model_rcm`
- `unet_rep3_perfect_model_rcm`
- `miniunet_perfect_model_rcm`
- `unet_seed2_perfect_model_rcm`

### Runtime

Reference interpreter on Kraken:

- `/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1/bin/python`

Canonical output root:

- `/scratch/globc/page/idownscale_output`

## 3. How to run

### Standalone workflow

Example:

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

Kraken submitter:

```bash
sbatch bin/production/submit_exp5_perfect_model_kraken.sh
```

### Important runtime path

Use:

```bash
export IDOWNSCALE_OUTPUT_DIR=/scratch/globc/page/idownscale_output
```

and not a repo-local fallback output tree.

## 4. Current validated outputs

Main comparison outputs:

- combined table:
  - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.csv`
  - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.md`
- score plot:
  - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_method_comparison_perfect_model_rcm_rcm.png`
- PDF / distribution plot:
  - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_distribution_pdf_perfect_model_rcm_rcm_tas.png`

Additional diagnostics:

- climate signal:
  - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_climate_signal_perfect_model_rcm_rcm_19810101_20101231_vs_20800101_21001231.csv`
- window statistics:
  - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_window_statistics_perfect_model_rcm_rcm.csv`

## 5. Current evaluation coverage

Direct prediction-vs-truth comparison currently validated in the combined table:

- `20000101_20141231`
- `20900101_21001231`

Climate-signal and variability diagnostics currently use:

- reference: `19810101_20101231`
- future: `20800101_21001231`

This is already enough to assess:

- historical performance
- late-century performance
- first-order climate-change signal fidelity
- variability behavior across windows

## 6. Results

### 6.1 Historical window (`2000-2014`)

Raw coarse-resolution input:

- bias: `0.338547 K`
- RMSE: `1.773409 K`

BC baselines:

- IBICUS CDFt:
  - bias: `0.258485 K`
  - RMSE: `1.855405 K`
- SBCK CDFt:
  - bias: `0.258475 K`
  - RMSE: `1.855406 K`

Best ML methods:

- `UNet + output norm`
  - bias: `-0.024973 K`
  - RMSE: `0.408541 K`
- `UNet cold replicate`
  - bias: `-0.033083 K`
  - RMSE: `0.451416 K`
- `MiniUNet`
  - bias: `0.139126 K`
  - RMSE: `0.477624 K`

Interpretation:

- ML strongly improves on both raw coarse input and BC
- in this perfect-model setup, the two CDFt implementations are essentially
  indistinguishable in the historical comparison

### 6.2 Late-century window (`2090-2100`)

Raw coarse-resolution input:

- bias: `0.301807 K`
- RMSE: `1.681219 K`

BC baselines:

- IBICUS CDFt:
  - bias: `0.230943 K`
  - RMSE: `1.760735 K`
- SBCK CDFt:
  - bias: `0.201856 K`
  - RMSE: `1.775895 K`

Best ML methods:

- `UNet + output norm`
  - bias: `-0.037046 K`
  - RMSE: `0.452707 K`
- `UNet cold replicate`
  - bias: `-0.071052 K`
  - RMSE: `0.479083 K`
- `MiniUNet`
  - bias: `0.148576 K`
  - RMSE: `0.506672 K`

Interpretation:

- ML still strongly improves on both raw coarse input and BC in late century
- IBICUS and SBCK remain close, but in the current setup SBCK is slightly worse
  than IBICUS on RMSE for `2090-2100`
- the ranking of ML methods remains stable, with `UNet + output norm` still the
  best all-around option

### 6.3 Climate-change signal

Current climate-signal comparison:

- reference: `1981-2010`
- future: `2080-2100`

Truth warming:

- `4.007120 K`

Current signal diagnostics:

- raw coarse input:
  - signal bias: `-0.046903 K`
  - signal RMSE: `0.143515 K`
  - correlation: `0.965594`
- default BC baseline:
  - signal bias: `-0.035465 K`
  - signal RMSE: `0.159253 K`
  - correlation: `0.954662`
- SBCK CDFt baseline:
  - signal bias: `-0.068613 K`
  - signal RMSE: `0.185034 K`
  - correlation: `0.943932`
- `UNet + output norm`:
  - signal bias: `-0.077079 K`
  - signal RMSE: `0.113563 K`
  - correlation: `0.987848`
- `MiniUNet`:
  - signal bias: `-0.056655 K`
  - signal RMSE: `0.100773 K`
  - correlation: `0.988320`

Interpretation:

- ML improves the spatial climate-change signal fidelity relative to both the
  raw coarse input and both BC baselines
- both BC paths preserve the broad warming tendency, but in the current setup
  neither BC implementation beats the raw coarse input on signal RMSE
- SBCK CDFt is slightly worse than the default CDFt path for the current
  climate-signal diagnostic

### 6.4 Variability diagnostics

The all-window statistics table now includes:

- mean bias
- RMSE
- mean absolute error
- day-to-day variability bias
- spatial variability bias
- annual variability bias

Main reading:

- raw and BC baselines both retain a strong negative spatial-variability bias
  across windows
- ML methods reduce that bias very substantially
- `UNet + output norm` remains the best balanced method across mean error,
  RMSE, and variability behavior
- IBICUS and SBCK are nearly identical up to mid-century; SBCK becomes slightly
  worse than the default BC path in the later future windows

## 7. Main scientific interpretation

At this stage the message is clear.

1. The perfect-model workflow is operational and scientifically usable.
2. ML clearly beats the degraded coarse-resolution RCM input.
3. ML also beats the BC baselines in the present perfect-model benchmark.
4. The two tested CDFt implementations, IBICUS and SBCK, are very close here.
   SBCK does not currently overturn the main conclusion.
5. `UNet + output norm` remains the best overall candidate because it combines:
   - the lowest or near-lowest RMSE
   - very small mean bias
   - good climate-signal fidelity
   - good variability behavior

## 8. Next scientific extension

The next natural extension is to broaden the same protocol to:

- other atmospheric variables
- other ML methods
- other RCM references
- additional BC methods when needed

The workflow is now close to that objective because:

- BC is handled as a selectable workflow component
- evaluation can compare several BC baselines side by side
- the diagnostics are no longer tied only to one ML model and one BC path
