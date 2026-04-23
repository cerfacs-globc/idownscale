#!/bin/bash
#SBATCH --job-name=CERT_AUDIT
#SBATCH --output=slurm_logs/cert_audit_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=grace
#SBATCH --time=00:15:00

set -e
export PYTHONPATH=$PYTHONPATH:.

echo "--- Final Certification: 20:48 CEST Baseline (0b7c3ec) ---"
echo "Date: $(date)"

# 1. Clean snapshots
rm -rf /scratch/globc/page/idownscale_output/audit_month/p1/*

# 2. Phase 1 Reconstruction for Jan 1st
./bin/run_grace.sh bin/preprocessing/build_dataset.py \
    --exp exp5 \
    --start_date 1980-01-01 \
    --end_date 1980-01-01 \
    --audit_dir /scratch/globc/page/idownscale_output/audit_month/p1

# 3. Parity Forensic
./bin/run_grace.sh /scratch/globc/page/idownscale_rerun/scratch/verify_jan01_authenticated.py

echo "--- Certification Complete ---"
