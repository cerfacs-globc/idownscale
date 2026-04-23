#!/bin/bash
#SBATCH --job-name=CERT_BENCHMARK_JAN
#SBATCH --output=slurm_logs/cert_bench_v86_%j.out
#SBATCH --error=slurm_logs/cert_bench_v86_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=grace
#SBATCH --time=01:00:00

# Climate Stabilization & Performance Benchmark (v86.74)
# Full month (Jan 1980) profile on Grace ARM environment
set -e

echo "--- Climate Stabilization Monthly Benchmark: Jan 1980 Cycle ---"
echo "Date: $(date)"
echo "Base Commit: $(git rev-parse --short HEAD)"

START_DATE="19800101"
END_DATE="19800131"
AUDIT_DIR="/scratch/globc/page/idownscale_output/audit_month"

mkdir -p $AUDIT_DIR/p1 $AUDIT_DIR/p2
mkdir -p slurm_logs

# --- Step 1: Phase 1 Reconstruction (Performance Profile) ---
echo ""
echo "--- Step 1: Phase 1 Reconstruction (31 Days) ---"
time ./bin/run_grace.sh bin/preprocessing/build_dataset.py \
    --exp exp5_audit \
    --start_date $START_DATE --end_date $END_DATE \
    --output_dir $AUDIT_DIR/p1

# --- Step 2: Phase 2 Bias Correction (Performance Profile) ---
echo ""
echo "--- Step 2: Phase 2 Bias Correction (31 Days) ---"
time ./bin/run_grace.sh bin/preprocessing/build_dataset_bc.py \
    --exp exp5 \
    --simu gcm \
    --start_date $START_DATE --end_date $END_DATE \
    --output_dir $AUDIT_DIR/p2

# --- Step 3: Scientific Parity Diagnostic (Monthly Stability) ---
echo ""
echo "--- Climate Stabilization Diagnostic (Jan 31 Audit) ---"
./bin/run_grace.sh bin/verification/certified_audit_v86.py

echo ""
echo "--- Benchmark Complete ---"
