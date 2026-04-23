#!/bin/bash
#SBATCH --job-name=audit_p1
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=00:20:00
#SBATCH --output=audit_p1_%j.out

# SBATCH logic followed by environment hardening
unset PYTHONHOME
unset PYTHONPATH
PYTHON_ARM=/softs/local_arm/Anaconda/2024.02-1/envs/gloenv_py3.11_arm/bin/python

export IDOWNSCALE_RAW_DIR=/gpfs-calypso/home/globc/page/idownscale_rerun/rawdata
export IDOWNSCALE_OUTPUT_DIR=/scratch/globc/page/idownscale_output
AUDIT_DIR=/scratch/globc/page/idownscale_output/audit

mkdir -p $AUDIT_DIR

echo "--- PHASE 1 AUDIT: RECONSTRUCTION (FRANCE) ---"
$PYTHON_ARM bin/preprocessing/build_dataset.py \
    --exp exp5 \
    --start_date 1980-01-20 \
    --end_date 1980-01-20 \
    --audit_dir $AUDIT_DIR

echo "--- VERIFYING BIT-PARITY (PHASE 1) ---"
$PYTHON_ARM bin/verification/verify_bit_parity.py \
    --new $AUDIT_DIR/sample_19800120.npz \
    --archival /scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/sample_19800120.npz \
    --label "Phase 1 Reconstruction AUDIT"
