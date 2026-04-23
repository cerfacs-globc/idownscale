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

set -e
module purge
module load python/gloenv3.12_arm

# Environment Synchronization (Production Baseline v86.74)
unset PYTHONHOME
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages:/scratch/globc/page/lib_idownscale_phase2/lib64/python3.12/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1

echo "--- 120-Year Climate Downscaling Production Pipeline (v86.74 Master) ---"
echo "Date: $(date)"
echo "Target: France (Phase 1) | Euro-Cordex (Phase 2)"
echo "Window: 1980-01-01 to 2100-12-31"

# 1. Industrial Sanitization
echo "--- Step 0: Sanitizing Production Directories ---"
mkdir -p /gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/
mkdir -p /gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_bc/
mkdir -p /scratch/globc/page/idownscale_output/weights/

# 2. Phase 1 Reconstruction (120 Years)
echo "--- Step 1: Phase 1 High-Resolution Reconstruction (120 Years) ---"
# Note: We use exp5 for France domain production.
python3 bin/preprocessing/build_dataset.py \
    --exp exp5 \
    --start_date 1980-01-01 \
    --end_date 2100-12-31 \
    --output_dir /gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y/

# 3. Phase 2 Bias Correction Synthesis (120 Years)
echo "--- Step 2: Phase 2 Bias Correction Synthesis (3 Periods + Master) ---"
# Note: This handles Train_Hist, Test_Hist, and Test_Future sequentially.
python3 bin/preprocessing/build_dataset_bc.py \
    --simu gcm \
    --exp exp5 \
    --var tas \
    --ssp ssp585 \
    --output_dir /gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_bc/

echo "--- 120-YEAR PRODUCTION SYNTHESIS COMPLETE ---"
date
