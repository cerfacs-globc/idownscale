#!/bin/bash
#SBATCH --job-name=idownscale_master_v86
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# ------------------------------------------------------------------------------
#  production_master_v86.sh [STEP 2 RECOVERY]
#
#  Hardened for 2025-2100 synthesis using ERA5 Persistence Guard.
# ------------------------------------------------------------------------------

set -e
module purge
module load python/gloenv3.12_arm

# Environment Synchronization (Production Baseline v86.74)
unset PYTHONHOME
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages:/scratch/globc/page/lib_idownscale_phase2/lib64/python3.12/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1

echo "--- 120-Year Climate Downscaling Synthesis Recovery (v86.74) ---"
echo "Date: $(date)"
echo "Window: 1980-01-01 to 2100-12-31"

# Step 0: Ensure Directory Structure
mkdir -p /gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_bc/

# Step 1: [ALREADY COMPLETED] High-Resolution Reconstruction

# Step 2: Phase 2 Bias Correction Synthesis (Hardened v86.74)
echo "--- Step 2: Phase 2 Bias Correction Synthesis (120 Years) ---"
python3 bin/preprocessing/build_dataset_bc.py \
    --simu gcm \
    --exp exp5 \
    --var tas \
    --ssp ssp585 \
    --output_dir /gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_bc/

echo "--- SYNTHESIS RECOVERY COMPLETE ---"
date
