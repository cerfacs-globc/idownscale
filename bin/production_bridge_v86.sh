#!/bin/bash
#SBATCH --job-name=idownscale_bridge_v86
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# ------------------------------------------------------------------------------
#  production_bridge_v86.sh — Stage 3.0 Discretization Bridge (120 Years)
#
#  Applies Ibicus CDF-t and generates daily sample snapshots for U-Net.
# ------------------------------------------------------------------------------

set -e
module purge
module load python/gloenv3.12_arm

# Environment Synchronization (Production Baseline v86.74)
unset PYTHONHOME
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages:/scratch/globc/page/lib_idownscale_phase2/lib64/python3.12/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1

echo "--- Stage 3.0 Discretization Bridge (120 Years) ---"
echo "Date: $(date)"

# Create Discretization Directory
mkdir -p /gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_bc/dataset_exp5_test_gcm_bc/

# Execute hardened Verification/Discretization Bridge
python3 bin/preprocessing/bias_correction_ibicus.py \
    --exp exp5 \
    --simu gcm \
    --var tas \
    --ssp ssp585

echo "--- STAGE 3.0 DISCRETIZATION COMPLETE ---"
date
