#!/bin/bash
#SBATCH --job-name=idownscale_array
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1       
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/idownscale_array_%A_%a.out
#SBATCH --error=slurm_logs/idownscale_array_%A_%a.err
#SBATCH --array=0-49

# AGGRESSIVELY unset Anaconda/x86 variables to avoid 'encodings' mismatch
# and ensure we pick up the correct system modules on worker nodes.
unset PYTHONHOME
unset PYTHONPATH
export PYTHONHOME=
export PYTHONPATH=

module load python/gloenv3.12_arm

# 12,784 production days / 50 nodes = ~256 days per task
CHUNK_SIZE=256
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$(( (SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE ))

# Execute the smart parallel wrapper
python3 bin/preprocessing/build_dataset_parallel.py --exp exp5 --i_start ${START_IDX} --i_end ${END_IDX}
