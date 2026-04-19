#!/bin/bash
#SBATCH -p grace
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o slurm_logs/provision_lib_%j.out
#SBATCH -e slurm_logs/provision_lib_%j.err

set -e

echo "Starting Scientifically-Synchronized Phase 2 Provisioning on $(hostname) at $(date)"

# Environment Hardening (Proven stability)
unset PYTHONHOME
unset PYTHONPATH
export PYTHONHOME=
export PYTHONPATH=
export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1

# Load established Phase 1 modules
module load python/gloenv3.12_arm

# Path to the new scratch library
LIB_DIR="/scratch/globc/page/lib_idownscale_phase2"
rm -rf "$LIB_DIR"  # Clean purge to resolve version conflicts
mkdir -p "$LIB_DIR"

# 1. Install AI and Climate Stack with strict NumPy version pin
# We pin to 1.26.4 to match the gloenv3.12_arm base module
echo "Installing pip dependencies (pinned to NumPy 1.26.4) into $LIB_DIR..."
pip install --target="$LIB_DIR" \
    "numpy==1.26.4" \
    ibicus \
    pytorch-lightning \
    monai \
    tqdm \
    torchmetrics \
    timm \
    seaborn \
    pybind11

# 2. Build SBCK from source into targeted directory
echo "Building SBCK from source with local Eigen 3.4.0 headers..."
export PYTHONPATH="$LIB_DIR:$PYTHONPATH"

cd /scratch/globc/page/idownscale_rerun/utils/SBCK/python
# We use --prefix to install into our scratch lib
python3 setup.py install --prefix="$LIB_DIR" eigen="../../eigen-3.4.0"

echo "Scientifically-Synchronized Provisioning Complete at $(date)"
