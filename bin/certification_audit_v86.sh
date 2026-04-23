#!/bin/bash
#SBATCH --job-name=CERT_BENCH_V86
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/cert_bench_v86_%j.out
#SBATCH --error=slurm_logs/cert_bench_v86_%j.err

set -e
module purge
module load python/gloenv3.12_arm

# Environment setup (Standardized v86.74)
unset PYTHONHOME
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages:/scratch/globc/page/lib_idownscale_phase2/lib64/python3.12/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1

# Output directory for 31-day benchmark (v86 version)
AUDIT_DIR="/scratch/globc/page/idownscale_output/audit_month_v86"
mkdir -p $AUDIT_DIR/p1 $AUDIT_DIR/p2

echo "--- 31-Day Climate Stabilization Benchmark (Jan 1980) ---"
echo "Date: $(date)"

# STEP 1: Phase 1 Reconstruction (France Domain - 64x64)
# This generates 31 daily files
echo "--- Step 1: Phase 1 Reconstruction (31 Days) ---"
python3 bin/preprocessing/build_dataset.py --exp exp5 --start_date 1980-01-01 --end_date 1980-01-31 --output_dir $AUDIT_DIR/p1

# STEP 2: Phase 2 Bias Correction (Euro-Cordex Domain - 29x28)
# This generates a single monthly aggregate file
echo "--- Step 2: Phase 2 Bias Correction (31 Days) ---"
python3 bin/preprocessing/build_dataset_bc.py --simu gcm --exp exp5_audit --var tas --start_date 1980-01-01 --end_date 1980-01-31 --output_dir $AUDIT_DIR/p2

# STEP 3: Scientific Parity Audit (January 31st Certification)
echo "--- Step 3: Scientific Parity Audit ---"
python3 bin/verification/certified_audit_v86.py --p1_new $AUDIT_DIR/p1/sample_19800131.npz --p2_new $AUDIT_DIR/p2/bc_train_hist_gcm.npz --idx 30

echo "--- Benchmark Complete ---"
