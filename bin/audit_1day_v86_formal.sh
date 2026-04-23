#!/bin/bash
#SBATCH --job-name=AUDIT_1D_FORMAL
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/audit_1d_formal_%j.out
#SBATCH --error=slurm_logs/audit_1d_formal_%j.err

set -e
module purge
module load python/gloenv3.12_arm

# Environment sync with production
unset PYTHONHOME
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages:/scratch/globc/page/lib_idownscale_phase2/lib64/python3.12/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1

# Isolated Audit Directory
AUDIT_DIR="/scratch/globc/page/idownscale_output/audit_1day_formal"
mkdir -p $AUDIT_DIR/p1 $AUDIT_DIR/p2

echo "--- Formal Production Audit: January 1, 1980 ---"
echo "Date: $(date)"

# STEP 1: Full Production Reconstruction (Phase 1 - France Focus)
echo "--- Step 1: Phase 1 Reconstruction (1 Day) ---"
python3 bin/preprocessing/build_dataset.py --exp exp5 --start_date 1980-01-01 --end_date 1980-01-01 --output_dir $AUDIT_DIR/p1

# STEP 2: Full Production Synthesis (Phase 2 - Euro-Cordex Focus)
echo "--- Step 2: Phase 2 Bias Correction (1 Day) ---"
python3 bin/preprocessing/build_dataset_bc.py --simu gcm --exp exp5_audit --var tas --start_date 1980-01-01 --end_date 1980-01-01 --output_dir $AUDIT_DIR/p2

# STEP 3: Formal Scientific Diagnostic
echo "--- Step 3: Scientific Parity Audit ---"
python3 bin/verification/certified_audit_v86.py --p1_new $AUDIT_DIR/p1/sample_19800101.npz --p2_new $AUDIT_DIR/p2/bc_train_hist_gcm.npz

echo "--- Audit Complete ---"
