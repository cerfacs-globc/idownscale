Hyperparameters Reference
=========================

Training entrypoints use ``iriscc/hparams.py`` and the
``IRISCCHyperParameters`` class to turn experiment configuration plus CLI
choices into a concrete training run configuration.

Core Fields
-----------

The most important hyperparameters are:

* ``exp`` / ``exp_name``: experiment key from ``CONFIG``
* ``run_name``: training run identifier used under ``runs/<exp>/<run_name>``
* ``model``: model family such as ``unet`` or ``cddpm``
* ``learning_rate``
* ``batch_size``
* ``max_epoch``
* ``loss``
* ``dropout``
* ``seed``

Dataset And Statistics Fields
-----------------------------

These fields are especially important for scientific reproducibility:

* ``sample_dir``: packaged dataset used for training
* ``statistics_dir``: directory containing the normalization statistics used by
  the transforms
* ``channels`` and ``in_channels``: predictor-channel contract
* ``img_size``: target grid shape
* ``mask`` and ``fill_value``: masking behavior
* ``output_norm`` and ``output_range``: output normalization behavior

During training, ``train.py`` copies the source ``statistics.json`` into a run-
local normalization directory and then updates ``statistics_dir`` to point to
that run-owned copy. That is why inference can keep using the training
statistics even when it evaluates on a different packaged sample tree such as a
BC evaluation dataset.

CDDPM-Specific Fields
---------------------

The diffusion workflow adds:

* ``n_steps``
* ``min_beta``
* ``max_beta``
* ``scheduler_step_size``
* ``scheduler_gamma``

These are stored in the checkpoint hyperparameters and are therefore available
to downstream inference and provenance logic.

How Hyperparameters Relate To ``settings.py``
---------------------------------------------

``settings.py`` decides experiment-wide scientific defaults such as target
dataset, grid, cadence, and date windows. ``IRISCCHyperParameters`` then turns
that experiment definition into a trainable model configuration.

In practice:

* ``CONFIG[exp]`` selects the dataset geometry and channels
* ``IRISCCHyperParameters`` selects how the model is trained on that dataset
* the saved checkpoint carries the model-side contract into inference

Operational Advice
------------------

When auditing a run, inspect these files together:

* the relevant ``CONFIG`` entry in ``settings.py``
* the run ``hparams.yaml``
* the run-local ``statistics.json``
* the provenance JSON written by training and prediction

If any of those disagree on sample roots, statistics roots, or channel counts,
the run should be treated as suspicious until resolved.
