#!/bin/bash
#SBATCH --job-name=audit_phase2
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load python/gloenv3.12_arm

# Environment setup (Unattended)
export PAGER=cat
export TERM=dumb

AUDIT_DIR="/scratch/globc/page/idownscale_output/audit"
mkdir -p $AUDIT_DIR

echo "--- Phase 2: Bias-Correction Synthesis Surgical Audit ---"
echo "Date: 1980-01-20"
echo "Environment: gloenv3.12_arm"

# 1. Run Production Builder for one day
# Note: Use GCM simulation for Exp5 TAS
python3 -u bin/preprocessing/build_dataset_bc.py --simu gcm --exp exp5 --var tas --start_date 1980-01-20 --end_date 1980-01-20 --audit_dir $AUDIT_DIR

# 2. Verify all variables and pixels (Includes 12x12 -> 64x64 regridding)
python3 -u bin/verification/verify_bit_parity.py \
    --new "$AUDIT_DIR/audit_bc_19800120.npz" \
    --archival "/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/sample_19800120.npz" \
    --label "Phase 2 Bias-Correction Certification"

echo "--- Audit Complete ---"
