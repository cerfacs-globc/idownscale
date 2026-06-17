Settings Reference
==================

``iriscc/settings.py`` is the central registry for runtime paths, source
catalog metadata, experiment definitions, time windows, and output naming.
Most workflow surprises come from misunderstanding one of those layers, so it
is worth reading the file as a contract rather than as a bag of constants.

Runtime Roots
-------------

The first responsibility of ``settings.py`` is to resolve the main filesystem
roots:

* ``RAW_DIR``: where upstream raw datasets live
* ``OUTPUT_DIR``: where generated datasets, runs, predictions, metrics, and
  graphs are written
* ``RUNTIME_ROOT``: parent directory used when explicit overrides are not set

These roots can be overridden with environment variables such as
``IDOWNSCALE_RAW_DIR`` and ``IDOWNSCALE_OUTPUT_DIR``. The workflow scripts do
not need to be edited when moving between machines if these variables are set
correctly.

Source Catalog
--------------

``SOURCE_CATALOG`` describes each supported upstream source. The most important
fields are:

* ``kind``: observation, reanalysis, or model
* ``root``: resolved raw-data directory
* ``geometry``: which grid family the source uses
* ``data_type``: loader family used by ``Data``
* ``historical_pattern`` / ``scenario_pattern`` or yearly/glob patterns:
  filename discovery rules
* ``native_frequency`` and ``default_frequency``: native cadence and default
  workflow cadence
* ``daily_aggregation`` or ``aggregation``: how a finer native cadence is
  aggregated when the workflow runs at a coarser cadence
* ``bias_corrected_root``: where corrected NetCDF outputs are stored for model
  sources

The CERRA integration follows this mechanism: native data are 3-hourly, while
the current training workflow defaults to daily means through the source
metadata rather than through a hard-coded special case.

Experiment Configuration
------------------------

``CONFIG`` entries define one scientific experiment. Each entry typically
combines:

* target description: ``target``, ``target_source``, ``target_vars``
* input description: ``channels``, ``input_vars``, model source aliases
* geometry: ``shape``, ``domain``, optional target/grid interpolation settings
* workflow dates and scenario: historical windows, future start, ``ssp``
* cadence: optional ``training_frequency`` and ``prediction_frequency``
* phase-specific behavior: BC method, target preparation, sample packaging

For current observation-target workflows:

* ``exp5`` uses E-OBS as the target observational dataset
* ``expc`` uses CERRA as the target observational dataset
* ``perfect_model_rcm`` uses the perfect-model framework with dedicated
  evaluation datasets

Path And Naming Helpers
-----------------------

The helper functions near the bottom of ``settings.py`` are just as important
as the raw configuration dictionaries. In practice they define the stable file
contracts used across preprocessing, training, inference, and evaluation.

The most-used helpers are:

* ``get_bc_bundle_path``: experiment-specific BC dataset bundles
* ``get_bias_corrected_netcdf_path``: corrected NetCDF file naming
* ``get_bias_corrected_sample_dir``: packaged BC sample directories
* ``get_dataset_variant_dir``: packaged raw/variant sample directories
* ``get_evaluation_sample_dir``: evaluation sample-tree routing
* ``get_prediction_output_path``: prediction NetCDF naming
* ``get_value_metrics_output_path``: VALUE metrics output naming

If a workflow stage needs a runtime path, prefer using one of these helpers
instead of rebuilding the path locally.

Frequency Helpers
-----------------

The cadence helpers are now explicit:

* ``get_source_native_frequency``
* ``get_source_default_frequency``
* ``get_source_aggregation_method``
* ``get_experiment_training_frequency``
* ``get_experiment_prediction_frequency``
* ``build_time_range``
* ``format_sample_time_token``

This separation matters when the observational target is sub-daily, such as
CERRA, but the packaged training workflow remains daily.

Scientific Checklist
--------------------

Before launching a new experiment, verify these settings-level questions:

* Which target observational dataset is selected?
* Which model source is used for the coarse predictor?
* What are the historical and future windows?
* What cadence is native, what cadence is used for training, and what cadence
  is used for prediction?
* Where will corrected NetCDFs, packaged samples, predictions, and metrics be
  written?
* Does the experiment point to experiment-specific BC artifacts rather than a
  shared location?

If these answers are not clear from the resolved context printed by the
workflow scripts, stop and fix the configuration before launching production
jobs.
