#!/bin/bash
#SBATCH -p grace
#SBATCH --gres=gpu:0
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o slurm_logs/setup_phase2_%j.out
#SBATCH -e slurm_logs/setup_phase2_%j.err

set -e

echo "Starting Phase 2 Hardened Setup (VENV) on $(hostname) at $(date)"

# AGGRESSIVELY unset Anaconda/x86 variables to avoid 'encodings' mismatch
# This was the key to successful Phase 1 execution
unset PYTHONHOME
unset PYTHONPATH
export PYTHONHOME=
export PYTHONPATH=

# Load the established Phase 1 gloenv
module load python/gloenv3.12_arm

# 1. Create the Virtual Environment OUTSIDE the git repo
PHASE2_ENV="/scratch/globc/page/env_idownscale_phase2"

# Aggressively isolate from home directory/corrupted site-packages
export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1

echo "Initializing ARM-native venv at $PHASE2_ENV..."
rm -rf "$PHASE2_ENV"
python3 -m venv --system-site-packages "$PHASE2_ENV"

# Activate the environment
source "$PHASE2_ENV/bin/activate"

# 2. Install AI and Climate Stack (Directly into environment, NO --user)
echo "Installing pip dependencies into environment..."
pip install \
    ibicus \
    pytorch-lightning \
    monai \
    tqdm \
    torchmetrics \
    timm \
    seaborn

# 3. Install SBCK from source (Directly into environment, NO --user)
echo "Installing SBCK from source into environment..."
cd /scratch/globc/page/idownscale_rerun/utils/SBCK
python3 setup.py install

echo "Phase 2 Environment Recovery Complete at $(date)"
