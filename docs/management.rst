Workflow Management
===================

The **idownscale** pipeline is managed through an automated master script ``run_exp5_full.sh``. This script coordinates the execution of multiple preprocessing, training, and evaluation phases.

Running the Pipeline
--------------------

The pipeline is designed to be run as a workload manager (Slurm) batch script, but it can also be executed interactively.

**Interactive Execution:**

.. code-block:: bash

   ./run_exp5_full.sh

**Workload Manager (Slurm):**

.. code-block:: bash

   sbatch run_exp5_full.sh

Execution Controls
------------------

The script supports environment variables for granular control:

* **START_PHASE** & **STOP_PHASE**: Run a specific range of phases (e.g., ``START_PHASE=4 STOP_PHASE=4`` for Training only).
* **FORCE=1**: Bypasses the marker check (``.markers/`` directory) in the bash script. Scripts will still **resume** from the last available date if possible.
* **REGENERATE=1**: Forces a full data regeneration by passing the ``--force`` flag to individual Python scripts.

Monitoring & Logging
--------------------

When monitoring long-running jobs, look for these healthy patterns:

Fast-Forwarding (Resumability)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If internal samples already exist, the scripts will skip the heavy interpolation work. Logs will show "Processing date..." at a very high rate. This is the expected behavior for resumed jobs.

Silent Computation Phases
^^^^^^^^^^^^^^^^^^^^^^^^^^

Some steps are computationally intensive and natively silent:

* ``compute_statistics_gamma.py``: Fits Gamma distributions for every pixel.
* ``bias_correction_ibicus.py``: The debiasing loop can be silent for long periods.

Customizing for Your Cluster (Example)
--------------------------------------

If you are using a different cluster or partition, update the ``#SBATCH`` headers in the script. For example, on a cluster with a dedicated CPU partition:

.. code-block:: bash

   # Example: Using the Grace partition (CPU-only)
   #SBATCH -p grace        
   #SBATCH --gres=gpu:0
