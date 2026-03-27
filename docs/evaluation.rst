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

   python3 bin/evaluation/compute_test_metrics_day.py --exp exp5 --test-name unet --startdate 20150101 --enddate 21001231

Score Visualization
-------------------

You can compare multiple models or simulations using boxplots:

.. code-block:: bash

   python3 bin/evaluation/compare_test_metrics.py --exp exp5 --test-list unet_gcm,unet_gcm_bc --scale monthly

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

The ``compute_value_metrics.py`` script provides a consolidated summary table based on the VALUE framework, comparing the downscaled model against ERA5 ground truth for the historical test period (2000-2014).

.. code-block:: bash

   python3 bin/evaluation/compute_value_metrics.py --exp exp5 --test-name unet_all --simu-test gcm_bc

For more information on the validation standards, visit the `VALUE website <http://www.value-cost.eu/>`_.
