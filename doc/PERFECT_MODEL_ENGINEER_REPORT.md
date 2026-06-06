# Perfect-Model Experiment Report

Date: 2026-06-05  
Platform: `kraken`  
Experiment: `perfect_model_rcm`

## 1. Purpose

This experiment implements a perfect-model evaluation workflow for statistical
downscaling.

Current scientific setup:

- pseudo-truth: native-resolution RCM output
- coarse predictor: the same RCM degraded to a coarser bridge grid
- baseline methods:
  - raw coarse-resolution RCM input
  - BC baseline
- ML methods:
  - `unet_outputnorm_perfect_model_rcm`
  - `unet_perfect_model_rcm`
  - `unet_rep3_perfect_model_rcm`
  - `miniunet_perfect_model_rcm`
  - `unet_seed2_perfect_model_rcm`

The goal is to answer:

- does ML improve over the coarse RCM input?
- does ML improve over BC alone?
- does ML preserve the climate-change signal?

## 2. Technical implementation

### Workflow

The standalone workflow runner is:

- `bin/production/run_exp5_perfect_model.py`

It now supports an explicit perfect-model chain including:

- BC dataset build
- BC application
- train dataset build
- eval dataset build
- train/eval sample validation
- dataset statistics
- training
- prediction
- prediction vs pseudo-truth comparison
- aggregate comparison
- score plotting
- PDF/distribution plotting

### Core scripts

Main validation and diagnostic scripts:

- `bin/evaluation/validate_perfect_model_samples.py`
- `bin/evaluation/compare_perfect_model_predictions_vs_truth.py`
- `bin/evaluation/aggregate_perfect_model_comparisons.py`
- `bin/evaluation/plot_perfect_model_comparison.py`
- `bin/evaluation/plot_perfect_model_distribution_pdf.py`
- `bin/evaluation/compare_perfect_model_climate_signal.py`

### Platform

Kraken submitter:

- `bin/production/submit_exp5_perfect_model_kraken.sh`

Reference Python interpreter used on Kraken:

- `/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1/bin/python`

Canonical output root:

- `/scratch/globc/page/idownscale_output`

## 3. How to run

### Full standalone workflow

Example Kraken-style invocation:

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

### Typical Slurm path on Kraken

```bash
sbatch bin/production/submit_exp5_perfect_model_kraken.sh
```

### Important runtime notes

- use the shared output root, not the repo-local fallback tree:
  - `IDOWNSCALE_OUTPUT_DIR=/scratch/globc/page/idownscale_output`
- this workflow now supports variable-aware prediction metadata through
  `predict_loop.py --var`
- the perfect-model ML path is aligned with the BC workflow so it mirrors
  `exp5` more closely

## 4. Current validated outputs

Main regenerated outputs:

- combined comparison table:
  - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.csv`
  - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_predictions_vs_truth_perfect_model_rcm_combined_rcm.md`
- score plot:
  - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_method_comparison_perfect_model_rcm_rcm.png`
- PDF/distribution plot:
  - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_distribution_pdf_perfect_model_rcm_rcm_tas.png`
- climate-signal comparison:
  - `/scratch/globc/page/idownscale_output/metrics/perfect_model_rcm/comparison_tables/perfect_model_climate_signal_perfect_model_rcm_rcm_20000101_20141231_vs_20900101_21001231.csv`
  - `/scratch/globc/page/idownscale_output/graph/metrics/perfect_model_rcm/perfect_model_climate_signal_perfect_model_rcm_rcm_20000101_20141231_vs_20900101_21001231.png`

## 5. Current scientific coverage

The currently regenerated combined comparison table contains:

- historical evaluation window:
  - `20000101_20141231`
- late-century future window:
  - `20900101_21001231`

This is enough for:

- historical performance comparison
- late-century performance comparison
- first climate-signal assessment

It is not yet the final desired scientific coverage for all future periods.

## 6. Results

### Late-century performance (`2090-2100`)

Raw coarse input and BC baseline in the final combined table:

- bias: `0.230943 K`
- RMSE: `1.760735 K`

ML results:

- `UNet + output norm`
  - bias: `-0.037046 K`
  - RMSE: `0.452707 K`
- `UNet`
  - bias: `-0.181476 K`
  - RMSE: `0.564187 K`
- `UNet replicate`
  - bias: `-0.055164 K`
  - RMSE: `0.601793 K`
- `MiniUNet`
  - bias: `0.148576 K`
  - RMSE: `0.506672 K`
- `UNet cold replicate`
  - bias: `-0.071052 K`
  - RMSE: `0.479083 K`

Interpretation:

- ML clearly outperforms the raw coarse-resolution RCM input
- ML also clearly outperforms the BC baseline in this current perfect-model
  comparison
- `UNet + output norm` remains the best overall compromise between low RMSE
  and small bias
- `MiniUNet` is competitive in RMSE but shows a warmer mean bias than the
  output-normalized UNet

### Climate-change signal

Current climate-signal comparison:

- reference window:
  - `20000101_20141231`
- future window:
  - `20900101_21001231`

Truth signal mean:

- `3.619777 K`

Signal metrics:

- raw coarse input:
  - signal RMSE: `0.160099 K`
  - signal correlation: `0.965484`
- BC baseline:
  - signal RMSE: `0.160117 K`
  - signal correlation: `0.965477`
- `UNet + output norm`:
  - signal bias: `-0.012073 K`
  - signal RMSE: `0.079900 K`
  - signal correlation: `0.991610`
- `MiniUNet`:
  - signal bias: `0.009450 K`
  - signal RMSE: `0.075264 K`
  - signal correlation: `0.992433`

Interpretation:

- ML improves climate-signal fidelity relative to both raw coarse input and BC
- BC baseline preserves the broad warming tendency, but in the current result
  it does not improve on the raw coarse input for signal RMSE

## 7. Current interpretation

Main conclusion at this stage:

- the perfect-model workflow is operational on Kraken
- BC is now integrated as a proper baseline in the perfect-model comparison
- ML still provides the strongest improvement over both raw coarse input and
  BC baseline
- the output-normalized UNet is currently the strongest all-around candidate

## 8. Recommended next scientific extension

The next evaluation step should be broadened beyond the current two-window
comparison.

Recommended climate-signal definition:

- reference:
  - `1981-2010`
- future:
  - `2080-2100`

Reason:

- `1981-2010` is the former climatological normal period
- `1991-2020` would be the more modern normal, but this RCM historical segment
  ends in `2014`

Recommended evaluation scope beyond the climate-signal panel:

- evaluate all available periods, not only historical and late-century
- add diagnostics for:
  - day-to-day variability
  - spatial variability
  - annual variability
- align these perfect-model diagnostics with the kinds of validation used in
  real downscaling evaluation, so the experiment is easier to compare to the
  non-perfect-model workflow
