Release Notes
=============

v1.1.0 - Stabilization & Research Readiness
-------------------------------------------

Summary
~~~~~~~
This version marks a critical milestone in the stabilization of the Experiment 5 pipeline. It restores scientific accuracy, introduces a comprehensive validation framework, and optimizes the repository for collaborative research.

Scientific Integrity
~~~~~~~~~~~~~~~~~~~~
* **Temperature Bias Resolution**: Fixed a major regression in Experiment 5 that caused a -180K bias.
* **Current status**: Mean bias is now **0.12 K** with a Spatial RMSE of **0.74 K**.
* **Loss Function**: Defaulted back to the stable ``masked_mse`` for EXP5.

Infrastructure & Automation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Automated Integrity Checks**: New ``bin/utils/check_pipeline_integrity.py`` performs deep health checks at every stage.
* **Structured Logging**: All process logs (stdout/stderr) are now routed to ``logs/<EXP>/<TIMESTAMP>/``.
* **Output Consolidation**: All validation artifacts are centralized in ``output/<EXP>/validation/``.
* **Workspace Portability**: ``bin/utils/setup_workspace.sh`` automates path remapping for new environments.

Documentation
~~~~~~~~~~~~~
* **Sphinx/RTD Support**: Full integration of release notes and validation protocols into the official documentation.
* **Unified Readme**: Updated with the new logging and validation structure.
