#!/bin/bash
#SBATCH --job-name=audit_phase1
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

echo "--- Phase 1: High-Resolution Reconstruction Surgical Audit ---"
echo "Date: 1980-01-20"
echo "Environment: gloenv3.12_arm"

# 1. Run Production Builder for one day
python3 -u bin/preprocessing/build_dataset.py --exp exp5 --start_date 1980-01-20 --end_date 1980-01-20 --audit_dir $AUDIT_DIR

# 2. Verify all variables and pixels
python3 -u bin/verification/verify_bit_parity.py \
    --new "$AUDIT_DIR/sample_19800120.npz" \
    --archival "/scratch/globc/page/idownscale_exp5/datasets/dataset_exp5_30y/sample_19800120.npz" \
    --label "Phase 1 Reconstruction Certification"

echo "--- Audit Complete ---"
