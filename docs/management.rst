The cleaned exp5 pipeline is managed through ``bin/production/run_exp5_workflow.py``.
It coordinates preprocessing and bias-correction steps with explicit skip or overwrite
behavior at the step level.

Running the Pipeline
--------------------

The workflow can be run on a workstation for small windows or on HPC for full production ranges.

**Interactive Execution (Debugging):**

.. code-block:: bash

   python bin/production/run_exp5_workflow.py --exp exp5 --steps phase1,stats

Optional downstream steps extend the same runner to Phase 3 and Phase 4 style tasks:

.. code-block:: bash

   python bin/production/run_exp5_workflow.py --exp exp5 --steps bc_dataset,bc_apply,raw_dataset
   python bin/production/run_exp5_workflow.py --exp exp5 --steps predict_loop,value_metrics --test-name unet_all --simu-test gcm_bc --predict-start-date <STARTDATE> --predict-end-date <ENDDATE> --value-start-date <STARTDATE> --value-end-date <ENDDATE>

**Workload Manager (Slurm):**

.. code-block:: bash

   sbatch path/to/your_local_workflow_submitter.sh

Production Certification & Parity
---------------------------------

To ensure scientific reproducibility, each production volume must be certified against a reference archival baseline.

* **Acceptance Criteria**: Historical parity aims for **0.00e+00 K** against the archival baseline.
* **Current State**: The clean branch tracks the remaining residuals and the forensic steps that reduced them.
* **Diagnostic Audit**: Surgical verification lives in ``bin/verification`` and the forensic notes in the repository root.

Execution Controls
------------------

The workflow runner uses explicit output existence checks:

* ``--if-exists skip``: Keep existing outputs and fast-forward completed steps.
* ``--if-exists overwrite``: Delete relevant outputs for the selected steps and rebuild them.

Path overrides
--------------

The clean branch allows path-level overrides through environment variables such as
``IDOWNSCALE_RUNS_DIR``, ``IDOWNSCALE_PREDICTION_DIR``, ``IDOWNSCALE_METRICS_DIR``,
and ``IDOWNSCALE_REGRID_WEIGHTS_DIR``. This is useful when prediction checkpoints
live somewhere other than the main ``OUTPUT_DIR`` tree.

Runtime path consistency
------------------------

The actively used prediction and evaluation entrypoints now share one runtime
resolution policy for:

* checkpoint discovery
* evaluation sample-directory lookup
* statistics-directory lookup

Operationally, this means that once ``RUNS_DIR`` and the runtime output roots
are configured correctly, the main scripts should agree on which checkpoint,
sample tree, and ``statistics.json`` belong to a given ``(exp, test_name,
simu_test)`` combination.

The same runtime helpers now also enforce file-discovery rules:

* checkpoint discovery must yield exactly one best checkpoint
* direct sample loads must point to an existing ``sample_<YYYYMMDD>.npz`` file
* helper logic that bootstraps metadata from sample trees must fail clearly if
  the directory is empty

When debugging a run, check these in order:

* the resolved-context block printed to stdout at startup
* the resolved checkpoint path
* the resolved sample directory
* the resolved statistics directory

If you need to force a different sample tree for a one-off rerun, prefer the
explicit CLI override (for example ``--sample-dir`` where supported) rather
than editing code or moving files in place.

Monitoring & Logging
--------------------

When monitoring long-running jobs, expect skip messages for already completed steps and
long silent stretches during regridding-heavy phases.

For Slurm-driven perfect-model production, prefer launching through local submit
wrappers and monitor the active job rather than assuming the wrapper defaults
match the intended scientific window.

Operationally important checks are:

* resolved-context stdout blocks at job start
* ``squeue`` or ``sacct`` state changes
* the final ``provenance_provjson=...`` line for each finished phase
* the presence of expected output files in the configured runtime roots

Customizing for Your Cluster
----------------------------

If you are using a different cluster or partition, adapt your local job wrapper rather than
hardcoding site assumptions into the Python workflow itself:

.. code-block:: bash

   # Example: Adjusting for your local partition
   #SBATCH -p my_partition        
   #SBATCH --gres=gpu:1
   #SBATCH --time=24:00:00
