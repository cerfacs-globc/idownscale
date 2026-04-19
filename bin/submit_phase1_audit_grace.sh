#!/bin/bash
#SBATCH --job-name=phase1_audit
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Phase 1 Isolation Audit (France Domain Regression)
# Verifies bit-parity for the foundational 30-year dataset

set -e

# --- Environment Setup ---
module purge
module load python/gloenv3.12_arm

unset PYTHONHOME
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages:/scratch/globc/page/lib_idownscale_phase2/lib64/python3.12/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1

echo "--- Phase 1 Isolation Audit: $(date) ---"
echo "Partition: Grace"
echo "Target: Experiment 5 (France Domain)"

python3 bin/verification/comprehensive_parity_audit.py

echo -e "\n--- Phase 1 Audit Complete ---"
