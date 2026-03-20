#!/bin/bash
#SBATCH -p grace
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp5_train
#SBATCH --output=exp5_train_%j.out
#SBATCH --error=exp5_train_%j.err

# Load ARM python module
module load python/anaconda3.11_arm

# Activate environment in scratch
CONDA_ENV_PATH="/scratch/globc/page/conda/envs/idownscale_env"
source activate $CONDA_ENV_PATH

# Run training
python bin/training/train.py
