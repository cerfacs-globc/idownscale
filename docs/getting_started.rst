Getting Started
===============

The **idownscale** project provides tools for climate downscaling that can run on
local workstations as well as HPC systems.

Installation
------------

We recommend a Conda-based setup because ``xesmf`` depends on ``ESMF``/``esmpy``.

.. code-block:: bash

   conda env create -f environment.yml
   conda activate idownscale
   pip install -e .

Runtime Configuration
---------------------

The main runtime paths are configured with environment variables:

.. code-block:: bash

   export IDOWNSCALE_RAW_DIR=/path/to/rawdata
   export IDOWNSCALE_OUTPUT_DIR=/path/to/output
   export IDOWNSCALE_REGRID_WEIGHTS_DIR=/path/to/output/weights
   export IDOWNSCALE_RUNS_DIR=/path/to/runs
   export IDOWNSCALE_PREDICTION_DIR=/path/to/prediction
   export IDOWNSCALE_METRICS_DIR=/path/to/metrics

Optional archival parity reference:

.. code-block:: bash

   export IDOWNSCALE_EXP5_ARCHIVE_DATASET_DIR=/path/to/archive/dataset_exp5_30y

For more detail, see ``README.md`` for the runtime workflow and
``doc/GRACE_TRAINING_ENGINEER_NOTE.md`` for the Grace-specific training route.

First Workflow Run
------------------

The cleaned exp5 entrypoint is:

.. code-block:: bash

   python bin/production/run_exp5_workflow.py --exp exp5 --steps phase1,stats --phase1-start-date 19850101 --phase1-end-date 19850103

On Grace, you can use the local wrapper:

.. code-block:: bash

   bash bin/production/run_exp5_workflow_grace.sh --exp exp5 --steps phase1,stats

Once coarse bias correction is built, the same runner can also package raw GCM test samples
and drive downstream prediction or VALUE evaluation steps if a trained checkpoint is available.

Grace GPU training
------------------

For Grace GPU training, the currently validated route is:

* modules:
  
  * ``python/gloenv3.12_arm``
  * ``nvidia/cuda/12.4``
* venv:
  
  * ``/scratch/globc/page/idownscale_envs/production_final_v22_312``

The longer engineering note is stored in ``doc/GRACE_TRAINING_ENGINEER_NOTE.md``.
