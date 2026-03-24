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
