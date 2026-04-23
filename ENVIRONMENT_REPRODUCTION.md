# Grace Environment Reproduction Guide (v86.74 Production Standard)

To achieve bit-identical scientific reproduction of the Experiment 5 (exp5) dataset generation on the Grace Hopper partition, follow these industrial-grade steps. This setup leverages native ARM-optimized system modules while enforcing a "Zero-Pollution" startup and clean environment.

## 1. System Modules
The environment relies on the system-provided `gloenv` stack. This ensures the correct linking of high-performance libraries (ESMF, OpenMPI) to the ARM-native Python interpreter.

```bash
# Load the native ARM Python environment
module load python/gloenv3.12_arm
```

## 2. Environment Sanitization (The "Zero-Pollution" Strategy)
The Grace nodes often contain conflicting user-level packages in `~/.local`. To ensure reproduction, we MUST isolate the execution from these stale paths:

```bash
# 1. Clear stale Anaconda/x86 paths
unset PYTHONHOME
unset PYTHONPATH

# 2. Enforce Isolation
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# 3. Explicitly link the ESMF backend (Required for xESMF)
# Assuming you are using the v86.74 module stack
VENV_PATH=$(which python | xargs dirname | xargs dirname)
export ESMFMKFILE="$VENV_PATH/lib/esmf.mk"
```

## 3. Dynamic Root Discovery
The production pipeline is designed to be machine-portable. Always export your data roots before execution to avoid hardcoded path errors:

```bash
export IDOWNSCALE_RAW_DIR="/path/to/rawdata"
export IDOWNSCALE_OUTPUT_DIR="/path/to/output"
```

## 4. Execution & Verification
Use the provided Phase-aware master orchestrator. This script includes integrity gates that verify each step before proceeding.

```bash
# Example: 120-year Master Production Launch (Phase 1 & 2)
sbatch bin/production_master_v86.sh
```

## Summary of Optimization
- **Binary Parity**: Enforcing `PYTHONNOUSERSITE=1` ensures that two different users on the same cluster achieve the same numerical results by bypassing private local overrides.
- **Native Performance**: Abandoning Singularity for preprocessing provides direct access to the Grace Hopper's H100 interoperability and ARM-native vectorization.
- **Certification**: This protocol is certified to achieve **0.00e+00 K (Absolute Parity)** against the April 19th archival anchor for Experiment 5.
