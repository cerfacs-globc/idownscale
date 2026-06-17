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

The main runtime paths are configured in ``iriscc/settings.py`` and can be
overridden with environment variables at runtime.

The most important overrides are:

.. code-block:: bash

   export IDOWNSCALE_RAW_DIR=/path/to/rawdata
   export IDOWNSCALE_OUTPUT_DIR=/path/to/output
   export IDOWNSCALE_GRAPHS_DIR=/path/to/graphs
   export IDOWNSCALE_REGRID_WEIGHTS_DIR=/path/to/output/regrid_weights
   export IDOWNSCALE_RUNS_DIR=/path/to/runs
   export IDOWNSCALE_PREDICTION_DIR=/path/to/prediction
   export IDOWNSCALE_METRICS_DIR=/path/to/metrics

Recommended separation when the repository lives in ``$HOME``:

.. code-block:: bash

   export IDOWNSCALE_RUNTIME_ROOT=/scratch/globc/$USER/idownscale_runtime
   export IDOWNSCALE_RAW_DIR=$IDOWNSCALE_RUNTIME_ROOT/rawdata
   export IDOWNSCALE_OUTPUT_DIR=$IDOWNSCALE_RUNTIME_ROOT/output
   export IDOWNSCALE_GRAPHS_DIR=$IDOWNSCALE_RUNTIME_ROOT/graphs
   export IDOWNSCALE_REGRID_WEIGHTS_DIR=$IDOWNSCALE_OUTPUT_DIR/regrid_weights

If ``IDOWNSCALE_RAW_DIR`` is not set explicitly, the code now uses:

1. ``repo/rawdata`` if that directory exists
2. otherwise ``$IDOWNSCALE_RUNTIME_ROOT/rawdata``

If ``IDOWNSCALE_OUTPUT_DIR`` is not set explicitly, the code now uses:

1. ``$IDOWNSCALE_RUNTIME_ROOT/output``

The repository also resolves evaluation, prediction, runs, and metrics roots
from the same settings layer. For production reruns, prefer explicit
environment overrides over editing scripts or hardcoding local paths.

Optional archival parity reference:

.. code-block:: bash

   export IDOWNSCALE_EXP5_ARCHIVE_DATASET_DIR=/path/to/archive/dataset_exp5_30y

For more detail, see:

* ``doc/ENVIRONMENT_SETUP.md``

First Workflow Run
------------------

The cleaned exp5 entrypoint is:

.. code-block:: bash

   python bin/production/run_obs_workflow.py --exp exp5 --steps phase1,stats --phase1-start-date 19850101 --phase1-end-date 19850103

If you are working on an HPC system and maintain local wrapper scripts, use
those wrappers instead of calling the Python entrypoint directly.

.. code-block:: bash

   sbatch path/to/your_local_workflow_submitter.sh

GPU is mainly useful for ``train`` and ``predict_loop``. Preprocessing and most
evaluation phases can run on CPU if GPU capacity is unavailable.

Once coarse bias correction is built, the same runner can also package raw GCM test samples
and drive downstream prediction or VALUE evaluation steps if a trained checkpoint is available.

Resolved Context and Provenance
-------------------------------

The main preprocessing, training, prediction, and workflow entrypoints now emit
two complementary provenance traces:

1. A resolved-context JSON block printed to stdout between:

   * ``=== IDOWNSCALE RESOLVED CONTEXT START ===``
   * ``=== IDOWNSCALE RESOLVED CONTEXT END ===``

2. A ``.prov.json`` file on disk, with its path echoed in stdout as
   ``provenance_provjson=...``.

These traces are meant to expose the values that have historically caused
silent workflow drift:

* settings-derived directories
* explicit start and end dates
* dataset and statistics paths
* checkpoint and run directories
* experiment and model names

For partial reruns, always pass explicit date windows. This is especially
important when running only prediction or comparison steps.

HPC note
--------

On some HPC systems, ``xesmf`` may require an additional environment variable
such as ``ESMFMKFILE`` so that ``ESMF`` / ``esmpy`` can be discovered correctly.
That path is installation-specific, so consult your local environment notes or
cluster documentation.

Typical pattern:

.. code-block:: bash

   export ESMFMKFILE=/path/to/esmf.mk
