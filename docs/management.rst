The **idownscale** pipeline is managed through the Grand Master Orchestrator ``bin/production/master_orchestrator_v86.sh``. This script coordinates the execution of multiple preprocessing, training, and evaluation phases with persistent progress tracking.

Running the Pipeline
--------------------

The pipeline is designed to be run as an automated batch job on high-performance compute clusters.

**Interactive Execution (Debugging):**

.. code-block:: bash

   ./bin/production/master_orchestrator_v86.sh

**Workload Manager (Slurm):**

.. code-block:: bash

   sbatch bin/production/master_orchestrator_v86.sh

Production Certification & Parity
---------------------------------

To ensure scientific reproducibility, each production volume must be certified against a reference archival baseline.

* **Master Census Protocol**: Run ``python3 bin/production/census_p1.py --exp exp5`` to verify bit-level parity.
* **Acceptance Criteria**: Parity verdicts must be exactly **0.00e+00 K** for historical volumes to be certified for research production.
* **Diagnostic Audit**: Surgical verification of coordinate shifts and metadata logic is performed via the provided audit scripts.

Execution Controls (Phase Markers)
----------------------------------

The orchestrator uses a persistent **Phase Marker** system to handle cluster node maintenance or preemption:

* **Marker Path**: ``.markers/phaseN.done`` tracks completion of each critical workflow step.
* **Resumability**: If a job is interrupted, the orchestrator will automatically skip completed phases.
* **Force Reset**: To rebuild a specific phase, manually delete the corresponding marker in the scratch directory.

Monitoring & Logging
--------------------

When monitoring long-running jobs, look for these healthy patterns:

Fast-Forwarding (Resumability)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If internal samples already exist, the scripts will skip the heavy interpolation work. This is the expected behavior for resumed jobs.

Silent Computation Phases
^^^^^^^^^^^^^^^^^^^^^^^^^^

Some steps are computationally intensive and natively silent:

* ``compute_statistics_gamma.py``: Fits Gamma distributions for every pixel.
* ``build_dataset_bc.py``: Regional regridding and bias correction.

Customizing for Your Cluster
----------------------------

If you are using a different cluster or partition, update the ``#SBATCH`` headers in the production script to match your local resource manager:

.. code-block:: bash

   # Example: Adjusting for your local partition
   #SBATCH -p my_partition        
   #SBATCH --gres=gpu:1
   #SBATCH --time=24:00:00
