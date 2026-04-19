#!/bin/bash
#SBATCH --job-name=idown_bc_synthesis
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load python/gloenv3.12_arm

# Environment setup (Matching certified audit)
unset PYTHONHOME
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages:/scratch/globc/page/lib_idownscale_phase2/lib64/python3.12/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1

echo "--- Phase 2: Ibicus Bias Correction Dataset Synthesis ---"
echo "Date: $(date)"
echo "Environment: gloenv3.12_arm"

# Run the synthesis for GCM-based predictors
# This processes DATES_BC_TRAIN_HIST, DATES_BC_TEST_HIST, and DATES_BC_TEST_FUTURE
python3 bin/preprocessing/build_dataset_bc.py --simu gcm --exp exp5 --var tas --ssp ssp585

echo "--- Synthesis Complete ---"
