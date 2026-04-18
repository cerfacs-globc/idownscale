# Grace Environment Reproduction Guide

To achieve scientific reproduction of the Experiment 5 (exp5) dataset generation on the Grace Hopper partition, follow these standard steps. This setup avoids complex Singularity containers for preprocessing and uses the native ARM-optimized system modules.

## 1. System Modules
The environment relies on the system-provided `gloenv` stack.

```bash
# Load the native ARM Python environment
module load python/gloenv3.12_arm
```

## 2. Environment Cleaning
The Grace nodes often have legacy Anaconda paths (x86) that can cause `ModuleNotFoundError: No module named 'encodings'`. Always clear these before execution:

```bash
unset PYTHONHOME
unset PYTHONPATH
export PYTHONHOME=
export PYTHONPATH=
```

## 3. Library Installation (Official Repos)
Follow the standard `SBCK` installation pattern (`pip install --user`) to ensure libraries are linked to the system's `ESMF 8.6.1` backend.

### ESMPy (Matching v8.6.1)
Instead of searching for elusive wheels, install the matching `esmpy` addon from the official ESMF repository:

```bash
git clone --depth 1 -b v8.6.1 https://github.com/esmf-org/esmf.git
cd esmf/src/addon/esmpy
python3 setup.py install --user
```

### xeSMF & Bias Correction
```bash
pip install xesmf ibicus --user
```

## 4. Execution
Use the provided `bin/run_grace.sh` wrapper which encapsulates the environment cleaning and module loading.

```bash
# Example: Generate dataset for Exp 5
bin/run_grace.sh python3 bin/preprocessing/build_dataset.py --exp exp5
```

## Summary of Optimization
- **Binary Compatibility**: Using `module load` ensures `xeSMF` links perfectly with the cluster's high-performance `ESMF` binaries.
- **Simplification**: Abandoning Singularity for preprocessing eliminates shared library (`libssl`, `libcurl`) conflicts while providing native ARM performance.
- **Rollback Safety**: This setup is additive to the user's home directory (`~/.local`) and does not modify system or repository files.
