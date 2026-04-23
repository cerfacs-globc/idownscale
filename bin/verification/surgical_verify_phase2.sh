#!/bin/bash
#SBATCH --job-name=verify_ph2_fast
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --time=00:05:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load python/gloenv3.12_arm

export PAGER=cat
export TERM=dumb

echo "--- Phase 2: Surgical Bit-Parity Verification ---"
python3 -u bin/verification/verify_bit_parity.py \
    --new "/scratch/globc/page/idownscale_output/audit/audit_bc_19800120.npz" \
    --archival "/scratch/globc/page/idownscale_exp5/datasets/dataset_bc/bc_train_hist_gcm.npz" \
    --label "Phase 2 Bias-Correction FINAL CERTIFICATION"

echo "--- Verification Complete ---"
