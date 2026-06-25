Evaluation
==========

The evaluation framework assesses the performance of downscaling models against reanalysis (SAFRAN/E-OBS) or high-resolution RCM data.

Metrics Calculation
-------------------

Metrics are computed at different temporal scales:

* **Daily Metrics**: ``compute_test_metrics_day.py``
* **Monthly Metrics**: ``compute_test_metrics_month.py``

To run evaluation for a specific experiment and model:

.. code-block:: bash

   python3 bin/evaluation/compute_test_metrics_day.py --exp exp5 --test-name unet --startdate <STARTDATE> --enddate <ENDDATE>

Runtime Input Resolution
------------------------

The main day, month, and VALUE evaluation entrypoints now share the same
runtime-path resolution logic used by the prediction tools.

For trained-model evaluation, the scripts resolve:

* the checkpoint from ``--checkpoint-bundle`` when provided, otherwise from the
  standard ``runs/<exp>/<test_name>/.../checkpoints`` location
* checkpoint discovery must now resolve to exactly one file; missing or
  ambiguous matches raise a clear error
* the statistics directory from checkpoint hyperparameters, preferring
  ``statistics_dir`` over ``sample_dir``
* the sample directory from the evaluation dataset mapping for
  ``(exp, test_name, simu_test)``, falling back to the training sample
  directory only when no evaluation mapping applies

For baseline and raw comparisons, the same sample-directory mapping is used
without requiring a checkpointed model.

For observation-target workflows, the default comparison suite now evaluates at
least three model-side methods whenever evaluation steps are requested:

* the raw driving model
* the bias-corrected baseline
* one or more selected ML methods

The corresponding comparison plot also includes the target/reference curve, so
the figure shows target + raw + BC + selected ML methods. This is configurable
through ``run_obs_workflow.py --compare-models ...`` and also works for BC-only
cases where no ML method is selected.

Requested daily sample files are also validated explicitly, so a missing
``sample_<YYYYMMDD>.npz`` now stops the run with the exact missing path.

This matters operationally because prediction and evaluation now resolve the
same sample roots by construction instead of duplicating slightly different

The older RCM-only metric scripts now use the same checked discovery policy for
their checkpoint, sample, and ALADIN target NetCDF inputs. Missing or
ambiguous matches therefore fail loudly instead of silently taking the first
glob hit.
logic in each script.

Score Visualization
-------------------

You can compare multiple models or simulations using boxplots:

.. code-block:: bash

   python3 bin/evaluation/compare_test_metrics.py --exp exp5 --test-list unet_gcm,unet_gcm_bc --scale monthly

Map plotting now distinguishes between:

* the data extent used to draw projected arrays
* the geographic plot extent used for the visible map window

When a plotting helper receives a geographic lon/lat domain, it now uses that
domain directly for the map view. For projected extents such as SAFRAN
``domain_xy``, the helper keeps the explicit France fallback viewport instead of
silently treating projected coordinates as lon/lat.

When that projected-domain fallback is used without an explicit
``plot_extent``, the helper now emits a warning so the chosen map viewport is
visible in logs and notebooks.

Future Trend Analysis
---------------------

The ``evaluate_future_trend.py`` script generates comprehensive maps and histograms comparing predictions with GCM/RCM future trends.

.. code-block:: bash

   python3 bin/evaluation/evaluate_future_trend.py --exp exp5 --ssp ssp585 --simu gcm

COST VALUE Framework
--------------------

The evaluation framework is aligned with the **COST Action ES1102 (VALUE)** validation standard for downscaling methods.
This approach assesses models across multiple dimensions to establish state-of-the-art performance:

* **Marginal Aspects**: Distributional accuracy via quantiles (q5, q50, q95) and variance ratios.
* **Temporal Aspects**: Lag-1 autocorrelation (persistence) and mean spell lengths.
* **Spatial Aspects**: Pattern correlation and spatial RMSE of climate variability maps.
* **Extremes**: Focus on high quantiles and distributional distances (Wasserstein distance).

Advanced Metrics Calculation
----------------------------

The ``compute_value_metrics.py`` script provides a consolidated summary table based on the VALUE framework, comparing the downscaled model against ERA5 ground truth for the configured historical validation window.

For wind-component pairs corrected jointly (for example ``uas``/``vas`` with
paired ``MBCn``), the standard scalar evaluation workflow should be applied to
derived wind speed once it has been materialized as ``sfcWind``. The helper
``bin/postprocessing/derive_wind_products.py`` derives:

* ``sfcWind`` from paired components
* ``windFromDirection`` in meteorological degrees clockwise from north

Only wind speed is directly compatible with the existing scalar metrics and
comparison suite. Wind direction is circular and therefore needs dedicated
direction-aware diagnostics rather than the default scalar VALUE/day/month
metrics.

.. code-block:: bash

   python3 bin/evaluation/compute_value_metrics.py --exp exp5 --test-name unet_all --simu-test gcm_bc

For more information on the validation standards, visit the `VALUE website <http://www.value-cost.eu/>`_.

Diagnostics assets and EGU-oriented materials are indexed in ``doc/DIAGNOSTICS_INDEX.md``.

Perfect-Model Validation
------------------------

The standalone perfect-model workflow has its own validation chain because the
reference is not an observational dataset but the native-resolution RCM
pseudo-truth packed in the sample ``y`` field.

Main utilities:

* ``bin/evaluation/validate_perfect_model_samples.py``
* ``bin/evaluation/compare_perfect_model_predictions_vs_truth.py``
* ``bin/evaluation/aggregate_perfect_model_comparisons.py``
* ``bin/evaluation/plot_perfect_model_comparison.py``
* ``bin/evaluation/plot_perfect_model_distribution_pdf.py``

Typical workflow:

.. code-block:: bash

   python bin/production/run_exp5_perfect_model.py \
     --exp perfect_model_rcm \
     --test-name unet_perfect_model_rcm \
     --steps build_train_dataset,build_eval_dataset,validate_train_dataset,validate_eval_dataset,stats,train,predict,compare_predictions,aggregate_comparison,plot_score_comparison,plot_distribution \
     --predict-startdate 20000101 \
     --predict-enddate 20141231 \
     --work-startdate 20000101 \
     --work-enddate 20141231

Validation is designed to fail early on:

* sample inventory mismatches
* incorrect ``x`` / ``y`` structure
* suspicious cross-period repetition in the coarse predictor
* distribution drift visible in the probability-density comparison

The explicit ``work-startdate`` and ``work-enddate`` arguments are important for
future-window or partial reruns. They prevent comparison and metrics steps from
silently falling back to the historical benchmark window.

Current validated outputs for the corrected ``perfect_model_rcm`` benchmark
include:

* a combined comparison table for historical and late-century windows
* a climate-signal comparison table and figure
* an all-window variability table
* method-comparison and distribution figures

Perfect-Model Provenance
------------------------

The perfect-model evaluation chain now writes workflow provenance files under:

* ``$IDOWNSCALE_OUTPUT_DIR/metrics/perfect_model_rcm/comparison_tables/``

In practice, the most useful provenance file is:

* ``workflow_<test_name>.prov.json``

Use it to confirm:

* which prediction file was scored
* which sample directory and statistics file were used
* which window was evaluated
* which settings-derived output roots were active

Where metadata are available in NetCDF outputs, these diagnostics prefer
``long_name`` / ``standard_name`` / ``units`` over hardcoded variable labels.
