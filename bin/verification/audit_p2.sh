#!/bin/bash
#SBATCH --job-name=audit_p2
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=00:20:00
#SBATCH --output=audit_p2_%j.out

# SBATCH logic followed by environment hardening
unset PYTHONHOME
unset PYTHONPATH
source /softs/Anaconda/2024.02-1/etc/profile.d/conda.sh
conda activate /softs/local_arm/Anaconda/2024.02-1/envs/gloenv_py3.11_arm

export IDOWNSCALE_RAW_DIR=/gpfs-calypso/home/globc/page/idownscale_rerun/rawdata
export IDOWNSCALE_OUTPUT_DIR=/scratch/globc/page/idownscale_output
AUDIT_DIR=/scratch/globc/page/idownscale_output/audit

mkdir -p $AUDIT_DIR

echo "--- PHASE 2 AUDIT: BIAS-CORRECTION (EUROPE) ---"
python bin/preprocessing/build_dataset_bc.py \
    --simu gcm \
    --exp exp5 \
    --start_date 1980-01-20 \
    --end_date 1980-01-20 \
    --audit_dir $AUDIT_DIR

echo "--- VERIFYING BIT-PARITY (PHASE 2) ---"
python bin/verification/verify_bit_parity.py \
    --new $AUDIT_DIR/audit_bc_19800120.npz \
    --archival /scratch/globc/page/idownscale_exp5/datasets/dataset_bc/bc_train_hist_gcm.npz \
    --label "Phase 2 Bias-Correction AUDIT"
