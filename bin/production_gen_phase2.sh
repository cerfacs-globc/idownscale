#!/bin/bash
#SBATCH --job-name=PROD_GEN_P2
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/prod_gen_p2_%j.out
#SBATCH --error=slurm_logs/prod_gen_p2_%j.err

set -e
module purge
module load python/gloenv3.12_arm
unset PYTHONHOME
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages:/scratch/globc/page/lib_idownscale_phase2/lib64/python3.12/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1

echo "--- STARTING PRODUCTION GENERATION: PHASE 2 (1980-2100) ---"
echo "Target: Euro Domain (exp5_audit)"
echo "Date: $(date)"

# Purge check
mkdir -p /gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_bc

python3 bin/preprocessing/build_dataset_bc.py --simu gcm --exp exp5_audit --var tas --ssp ssp585

echo "--- PHASE 2 GENERATION COMPLETE ---"
