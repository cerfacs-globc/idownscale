Notebook Examples
=================

These notebooks are lightweight operator guides rather than executed benchmark
artifacts. They show the command structure, validation points, and provenance
checks for the most common workflow variants.

Available notebooks
-------------------

* :download:`BC+ML tas downscaling with E-OBS and GCM <notebooks/tas_bc_ml_eobs_gcm.ipynb>`
* :download:`BC+ML tas downscaling with CERRA and GCM <notebooks/tas_bc_ml_cerra_gcm.ipynb>`
* :download:`BC-only tas workflow with E-OBS and GCM <notebooks/tas_bc_only_eobs_gcm.ipynb>`
* :download:`BC+ML tas downscaling with E-OBS and RCM <notebooks/tas_bc_ml_eobs_rcm.ipynb>`
* :download:`Perfect-model BC+CDDPM with RCM input <notebooks/perfect_model_bc_cddpm_rcm.ipynb>`

How To Use Them
---------------

Each notebook follows the same structure:

* resolve runtime roots and experiment choice
* show the main workflow commands
* list the expected outputs at each stage
* highlight the provenance files and validation checks to inspect

They are meant to be adapted to your runtime paths and Slurm wrappers rather
than executed unmodified on every machine.
