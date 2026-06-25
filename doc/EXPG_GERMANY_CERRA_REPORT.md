# Germany+CERRA `expg` report

Date: 2026-06-25  
Platform: `kraken`  
Experiment: `expg`

## 1. Objective

`expg` is the Germany-target observation workflow using CERRA as the final
target dataset and `tas` as the variable. The goal of this run was to validate
the full production chain on a non-France domain and to compare four paths:

- raw GCM baseline
- BC baseline
- ML on raw packaged inputs: `ML(raw)`
- ML on BC packaged inputs: `ML(BC)`

This comparison is important because earlier France workflows had made
`BC+ML` look like the default best path, while Germany+CERRA is more demanding
because of the broader domain, stronger continental structure, and more complex
topography.

## 2. Domain And Workflow

The corrected Germany domain used for the final run is a reduced rectangular
subset designed to avoid as much of the Alpine southeast as possible while
keeping north and east Germany:

- `[6.0, 14.8, 48.1, 55.2]`

The final workflow components were:

- target: daily CERRA over the Germany target box
- historical reference for the ML training pathway: ERA5 bridged to the target
  grid
- coarse model: `CNRM-CM6-1`
- BC reference: ERA5 on the BC/coarse workflow space
- ML model: `unet_outputnorm_expg_cerra_germany`

The key production lesson from this run is that BC must be assessed, not
assumed beneficial.

## 3. Main Outputs

Predictions:

- historical ML on BC:
  - `/scratch/globc/page/idownscale_runtime/output/prediction/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20000101_20210910_expg_unet_outputnorm_expg_cerra_germany_gcm_bc.nc`
- future ML on BC:
  - `/scratch/globc/page/idownscale_runtime/output/prediction/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20210911_21001231_expg_unet_outputnorm_expg_cerra_germany_gcm_bc.nc`
- historical ML on raw:
  - `/scratch/globc/page/idownscale_runtime/output/prediction/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20000101_20210910_expg_unet_outputnorm_expg_cerra_germany_gcm.nc`

Metrics:

- mean metrics:
  - `/scratch/globc/page/idownscale_runtime/output/metrics/expg/mean_metrics/`
- VALUE metrics:
  - `/scratch/globc/page/idownscale_runtime/output/metrics/expg/`
- plots:
  - `/scratch/globc/page/idownscale_runtime/graphs/metrics/expg/`

## 4. Historical Results

### Daily all-period metrics

- raw baseline:
  - temporal RMSE: `4.375245 K`
  - spatial bias: `-0.205402 K`
  - spatial RMSE: `5.704941 K`
- BC baseline:
  - temporal RMSE: `4.536219 K`
  - spatial bias: `-1.313831 K`
  - spatial RMSE: `6.151561 K`
- `ML(raw)`:
  - temporal RMSE: `4.360758 K`
  - spatial bias: `-0.903162 K`
  - spatial RMSE: `5.696095 K`
- `ML(BC)`:
  - temporal RMSE: `4.612736 K`
  - spatial bias: `-2.035072 K`
  - spatial RMSE: `6.226798 K`

### Monthly all-period metrics

- raw baseline:
  - temporal RMSE: `2.367457 K`
  - spatial bias: `-0.006690 K`
  - spatial RMSE: `0.556715 K`
- BC baseline:
  - temporal RMSE: `2.427653 K`
  - spatial bias: `-0.043582 K`
  - spatial RMSE: `0.606610 K`
- `ML(raw)`:
  - temporal RMSE: `2.321619 K`
  - spatial bias: `-0.029657 K`
  - spatial RMSE: `0.556192 K`
- `ML(BC)`:
  - temporal RMSE: `2.581450 K`
  - spatial bias: `-0.067296 K`
  - spatial RMSE: `0.640425 K`

### Winter behavior

Winter is the hardest season and shows the main degradation very clearly:

- raw monthly winter spatial bias: `-0.815623 K`
- BC monthly winter spatial bias: `-5.312994 K`
- `ML(raw)` monthly winter spatial bias: `-3.615423 K`
- `ML(BC)` monthly winter spatial bias: `-8.203930 K`

So the BC pathway is already strongly too cold in winter, and the ML on BC path
amplifies that drift further.

### VALUE metrics

- raw baseline:
  - bias: `-0.008230 K`
  - std ratio: `1.087043`
  - Wasserstein: `0.628995`
  - spatial correlation: `0.557189`
  - spatial RMSE: `0.768119 K`
- BC baseline:
  - bias: `-0.967793 K`
  - std ratio: `1.051446`
  - Wasserstein: `0.969429`
  - spatial correlation: `0.619654`
  - spatial RMSE: `1.179983 K`
- `ML(raw)`:
  - bias: `-0.601400 K`
  - std ratio: `1.076213`
  - Wasserstein: `0.818599`
  - spatial correlation: `0.791060`
  - spatial RMSE: `0.802562 K`
- `ML(BC)`:
  - bias: `-1.584014 K`
  - std ratio: `1.038760`
  - Wasserstein: `1.585884`
  - spatial correlation: `0.875060`
  - spatial RMSE: `1.637571 K`

## 5. Interpretation

The main result is not that the raw baseline is always “best”. The raw baseline
looks good in part because it is smoother than the target. The more relevant
scientific conclusion is:

- `ML(raw)` is the best ML configuration for `expg`
- `ML(BC)` is clearly degraded by the BC input
- the raw baseline remains a strong smooth reference, especially on mean bias

This means:

- BC is not justified automatically for this experiment family
- the production comparison must include raw, BC, `ML(raw)`, and `ML(BC)`
- the final production path should be chosen from the evidence, not from the
  assumption that more preprocessing is always better

The current best compromise for `expg` is therefore `ML(raw)`: it improves the
spatial structure relative to the raw baseline while avoiding the strong cold
drift introduced by the BC pathway.

## 6. Diagnostic Conclusion

The diagnostics done during this run indicate that:

- the ERA5-to-target-grid bridge used for the ML training pathway is good
- the degradation starts in the BC path
- the BC field is not obviously nonsensical on its native corrected grid
- the main problem appears after the corrected field is packaged/remapped into
  the final target-grid workflow

So the current scientific interpretation is that Germany+CERRA is a case where
the present `GCM -> ERA5` BC pathway does not help the final target-space ML
workflow. This is a real result, not a reason to force BC anyway.

## 7. Production Rule

For future observation-target production runs, the minimum assessment should
always include:

- raw baseline
- BC baseline
- `ML(raw)`
- `ML(BC)`

Operationally, that means evaluating at least:

- `--simu-test gcm`
- `--simu-test gcm_bc`

and selecting the final production path from the resulting diagnostics.
