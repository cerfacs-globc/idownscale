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
