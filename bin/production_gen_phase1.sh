#!/bin/bash
#SBATCH --job-name=PROD_GEN_P1
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/prod_gen_p1_%j.out
#SBATCH --error=slurm_logs/prod_gen_p1_%j.err

set -e
module purge
module load python/gloenv3.12_arm
unset PYTHONHOME
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages:/scratch/globc/page/lib_idownscale_phase2/lib64/python3.12/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1

echo "--- STARTING PRODUCTION GENERATION: PHASE 1 (1980-2014) ---"
echo "Target: France Domain (exp5)"
echo "Date: $(date)"

# Purge check
mkdir -p /gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y

python3 bin/preprocessing/build_dataset.py --exp exp5 --output_dir /gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_exp5_30y

echo "--- PHASE 1 GENERATION COMPLETE ---"
