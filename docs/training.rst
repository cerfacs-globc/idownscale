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

On Grace GPU nodes, we currently recommend using CSV logging instead of TensorBoard
for the first smoke or production training runs. The repository training entrypoint
supports this through the ``IDOWNSCALE_FORCE_CSV_LOGGER=1`` environment variable.

Grace GPU training
------------------

A known-good Grace GPU training route is now available.

Working combination:

* module stack:
  
  * ``python/gloenv3.12_arm``
  * ``nvidia/cuda/12.4``
* Python environment:
  
  * ``/scratch/globc/page/idownscale_envs/production_final_v22_312``
* recommended training flags:
  
  * ``IDOWNSCALE_FORCE_CSV_LOGGER=1``
  * ``IDOWNSCALE_SKIP_TEST_FIGURES=1``

Typical Grace submit pattern:

.. code-block:: bash

   sbatch --export=ALL,\
   TEST_NAME=unet_smoke,\
   STEPS=train,\
   IF_EXISTS=overwrite,\
   MAX_EPOCH=1,\
   IDOWNSCALE_VENV_PATH=/scratch/globc/page/idownscale_envs/production_final_v22_312,\
   IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES=tqdm,\
   IDOWNSCALE_FORCE_CSV_LOGGER=1,\
   IDOWNSCALE_SKIP_TEST_FIGURES=1 \
   bin/production/submit_exp5_train_grace.sh

The longer engineer-facing note with failure history and reconstruction advice is
stored in ``doc/GRACE_TRAINING_ENGINEER_NOTE.md``.

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

   # Loop prediction (e.g., full 2015-2100 period)
   python bin/training/predict_loop.py --startdate 20150101 --enddate 21001231 --exp exp5
