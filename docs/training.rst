Model Training
==============

The project uses PyTorch Lightning to manage training workflows, supporting multiple architectures and loss functions.

Training Process
----------------

Training is initiated via ``bin/training/train.py``. Configuration is managed through the ``IRISCCHyperParameters`` class in ``iriscc/hparams.py``.

.. code-block:: bash

   python bin/training/train.py

Training logs and model checkpoints are saved in the ``runs/`` directory. You can monitor progress using Tensorboard:

.. code-block:: bash

   tensorboard --logdir=runs/

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
