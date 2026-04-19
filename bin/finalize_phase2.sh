#!/bin/bash
#SBATCH -p grace
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o slurm_logs/finalize_phase2_%j.out
#SBATCH -e slurm_logs/finalize_phase2_%j.err

set -e

echo "Starting Unified Phase 2 Finalization on $(hostname) at $(date)"

# Environment Hardening (Proven in Phase 1)
unset PYTHONHOME
unset PYTHONPATH
export PYTHONHOME=
export PYTHONPATH=
export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1

# Load established Phase 1 modules
module load python/gloenv3.12_arm

# Activate the existing scratch-based Phase 2 environment
source "/scratch/globc/page/env_idownscale_phase2/bin/activate"

# 1. Install pybind11 natively
echo "Installing pybind11 natively..."
pip install pybind11

# 2. Build SBCK from source using local Eigen headers
echo "Building SBCK from source with local Eigen 3.4.0 headers..."
# The path to eigen from utils/SBCK/python is ../../eigen-3.4.0
cd /scratch/globc/page/idownscale_rerun/utils/SBCK/python
python3 setup.py install eigen="../../eigen-3.4.0"

# 3. Final Scientific Certification
echo "Performing Final Phase 2 Certification..."
python3 -c "import SBCK; import ibicus; import pytorch_lightning; print('PHASE 2 SCIENTIFIC STACK: CERTIFIED SUCCESS')"

echo "Phase 2 Finalization Complete at $(date)"
