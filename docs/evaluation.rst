Evaluation
==========

The evaluation framework assesses the performance of downscaling models against reanalysis (SAFRAN/E-OBS) or high-resolution RCM data.

Metrics Calculation
-------------------

Metrics are computed at different temporal scales to assess both short-term variability and long-term climate signals.

* **Daily Metrics**: ``compute_test_metrics_day.py`` computes pixel-wise errors (MAE, RMSE, Bias) on each day of the test set.
* **Monthly Metrics**: ``compute_test_metrics_month.py`` aggregates the daily predictions into monthly means before calculating errors. This is useful for identifying seasonal biases.

These scripts generate ``.csv`` or ``.json`` summary files in the ``metrics/`` directory.

Score Visualization
-------------------

You can compare multiple models or simulations using boxplots to visualize the distribution of errors across the domain:

.. code-block:: bash

   python3 bin/evaluation/compare_test_metrics.py --exp exp5 --test-list unet_gcm,unet_gcm_bc --scale monthly

Future Trend Analysis
---------------------

The ``evaluate_future_trend.py`` script is the most comprehensive evaluation tool. It compares:

1. **Raw Trend**: The temperature change projected by the original GCM/RCM (interpolated).
2. **Downscaled Trend**: The temperature change projected by the UNet model.
3. **Reference Change**: The spatial distribution of changes relative to the 1980-2010 baseline.

**Output Details:**
- **Spatial Maps**: 3x3 grid showing changes for three future periods (2015-2040, 2040-2070, 2070-2100).
- **Variability Boxplots**: Comparison of daily variability between the raw simulation and the downscaled model.
- **Histograms**: Shift in temperature distributions over time.

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

The ``compute_value_metrics.py`` script provides a consolidated summary table based on the VALUE framework, comparing the downscaled model against ERA5 ground truth for the historical test period (2000-2014).

.. code-block:: bash

   python3 bin/evaluation/compute_value_metrics.py --exp exp5 --test-name unet_all --simu-test gcm_bc

For more information on the validation standards, visit the `VALUE website <http://www.value-cost.eu/>`_.
