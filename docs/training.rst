Model Training
==============

The project uses PyTorch Lightning to manage training workflows, supporting multiple architectures and loss functions.

Training Process
----------------

Training is initiated via ``bin/training/train.py``. This script leverages **PyTorch Lightning** to handle the heavy lifting of the training loop:

* **Orchestration**: It manages device placement (CPU/GPU), precision (float32/float16), and the optimization steps.
* **Logging**: Real-time metrics (training loss, validation loss) are sent to the ``TensorBoardLogger``.
* **Checkpointing**: The ``ModelCheckpoint`` callback ensures that only the best model (based on validation loss) is saved in the ``runs/`` directory.

Hyperparameters (learning rate, batch size, model architecture) are managed through the ``IRISCCHyperParameters`` class in ``iriscc/hparams.py``.

.. code-block:: bash

   python bin/training/train.py

Training logs and model checkpoints are saved in the ``runs/`` directory. You can monitor progress using Tensorboard:

.. code-block:: bash

   tensorboard --logdir=runs/

Inference and Projections
-------------------------

Once trained, the model is used to downscale climate projections. This is handled by ``bin/training/predict_loop.py``.

**How it works:**
1. **Model Loading**: It loads the `best-checkpoint` from the specified experiment run.
2. **Sequential Prediction**: It iterates through the target dates (e.g., 2015-2100).
3. **Data Transformation**: For each day, it loads the bias-corrected GCM/RCM sample, applies the same normalization as training, and runs the forward pass through the UNet.
4. **NetCDF Reconstruction**: The individual daily predictions are reassembled into a standard NetCDF CF-compliant file.

.. code-block:: bash

   # Loop prediction (e.g., full 2015-2100 period)
   python bin/training/predict_loop.py --startdate 20150101 --enddate 21001231 --exp exp5 --test-name my_test --simu-test gcm_bc
