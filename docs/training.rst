Model Training
==============

The project uses PyTorch Lightning to manage training workflows, supporting multiple architectures and loss functions.

Training Process
----------------

Training is initiated via ``bin/training/train.py``. Configuration is managed through the ``IRISCCHyperParameters`` class in ``iriscc/hparams.py``.

.. code-block:: bash

   python bin/training/train.py

Training logs and model checkpoints are saved in the ``runs/`` directory.

On environments where TensorBoard is available, you can monitor progress with:

.. code-block:: bash

   tensorboard --logdir=runs/

On constrained or unusual environments, CSV logging can be safer than
TensorBoard for first smoke or production runs. The repository training
entrypoint supports this through the ``IDOWNSCALE_FORCE_CSV_LOGGER=1``
environment variable.

Cluster Training Notes
----------------------

For HPC or cluster training, the exact Python, CUDA, and remapping stack can be
site-specific.

Commonly useful flags:

* ``IDOWNSCALE_FORCE_CSV_LOGGER=1``
* ``IDOWNSCALE_SKIP_TEST_FIGURES=1``
* ``IDOWNSCALE_SKIP_TEST=1``

If your environment uses ``xesmf``, it may also require an installation-specific
``ESMFMKFILE`` setting for ``ESMF`` / ``esmpy``.

Typical pattern:

.. code-block:: bash

   export ESMFMKFILE=/path/to/esmf.mk

For the standalone perfect-model workflow, the repository launcher uses
``--skip-test`` during training and relies on the explicit prediction,
comparison, and validation phases afterwards. This avoids spending cluster time
on a redundant post-fit test pass.

Architectures
-------------

Supported models include:
* **UNet**: Standard encoder-decoder for spatial downscaling.
* **Swin2SR**: Transformer-based architecture for super-resolution.
* **CDDPM**: Conditional Denoising Diffusion Probabilistic Model.

Perfect-Model Framework
-----------------------

The standalone perfect-model framework is driven by
``bin/production/run_exp5_perfect_model.py``.

The corrected temperature workflow now uses the same scientific conditioning
logic for the full ML comparison set:

* BC baselines are evaluated explicitly as standalone methods
* ML models are trained in BC+ML mode
* CDDPM is conditioned on the same BC-informed predictor family as the UNet
  models

For ``perfect_model_rcm``, the input tensor contains elevation, degraded coarse
temperature, and bias-corrected coarse temperature. The target is the native
RCM temperature field.

Replicates used in the current benchmark:

* ``unet_rep3_perfect_model_rcm``: third UNet replicate with the same protocol
  and a different random initialization
* ``unet_seed2_perfect_model_rcm``: UNet replicate trained with seed 2

Inference
---------

To generate predictions for specific dates or long loops:

.. code-block:: bash

   # Single date prediction
   python bin/training/predict.py --date 20121018 --exp exp5

   # Loop prediction over an explicit configured window
   python bin/training/predict_loop.py --startdate <STARTDATE> --enddate <ENDDATE> --exp exp5

``predict_loop.py`` also supports ``--var`` and writes lightweight dataset
provenance attributes so downstream diagnostics can recover experiment, sample,
and source-role context from the prediction NetCDF itself.

Its time stepping now follows the experiment ``prediction_frequency`` setting.
For fixed-step sub-daily prediction, this means inference resolves
``sample_<YYYYMMDDHH>.npz`` inputs and emits a matching ``3h``/``6h``/``12h``
token in the prediction NetCDF filename instead of assuming daily timestamps.

Runtime Resolution Order
------------------------

Prediction entrypoints now use the same shared runtime-path resolution logic as
the main evaluation scripts.

For checkpoint resolution:

* if ``--checkpoint-bundle`` is passed, use the bundled checkpoint
* otherwise load the best checkpoint from
  ``$IDOWNSCALE_OUTPUT_DIR/runs/<exp>/<test_name>/lightning_logs/version_best/checkpoints/``
* if zero or multiple checkpoint matches are found, fail immediately with the
  searched directory and pattern instead of picking one silently

For sample-directory resolution:

* if an explicit ``--sample-dir`` is passed, use it
* otherwise, if ``test_name`` / ``simu_test`` map to an evaluation dataset
  variant, use that evaluation sample directory
* otherwise fall back to the training ``sample_dir`` stored in the checkpoint
  hyperparameters
* if no checkpoint-backed value exists, fall back to the experiment dataset in
  ``settings.py``

For statistics resolution:

* prefer ``statistics_dir`` from the checkpoint hyperparameters when present
* otherwise use the resolved sample directory

For sample-file access:

* prediction and evaluation entrypoints now check that each requested
  ``sample_<YYYYMMDD>.npz`` file exists before loading it
* shape-bootstrap logic such as perfect-model target discovery also fails
  clearly when a sample directory is empty

This shared resolution order reduces drift between prediction and evaluation
scripts and makes provenance easier to interpret.

For perfect-model reruns, pass explicit windows all the way through the
standalone launcher:

.. code-block:: bash

   python bin/production/run_exp5_perfect_model.py \
     --exp perfect_model_rcm \
     --test-name cddpm_perfect_model_rcm \
     --steps predict,compare_predictions \
     --predict-startdate 20900101 \
     --predict-enddate 21001231 \
     --work-startdate 20900101 \
     --work-enddate 21001231

The explicit ``work`` window is important because comparison and metrics steps
must use the same window as prediction.

Training and Inference Provenance
---------------------------------

Training and prediction stages now write both stdout resolved-context blocks and
``.prov.json`` files.

Common locations:

* ``$IDOWNSCALE_OUTPUT_DIR/runs/<exp>/<test_name>/provenance_train.prov.json``
* ``$IDOWNSCALE_OUTPUT_DIR/prediction/...<test_name>....prov.json``

The most useful fields are:

* ``parameters``: dates, test name, number of diffusion samples, output range
* ``settings``: dataset, evaluation dataset, runs dir, prediction dir
* ``inputs``: checkpoint, statistics file, sample directory
* ``outputs``: prediction NetCDF and metric tables
