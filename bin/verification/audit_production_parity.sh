#!/bin/bash
#SBATCH --job-name=audit_prod_parity
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
module purge
module load python/gloenv3.12_arm

export PYTHONPATH=$PYTHONPATH:.
python3 bin/preprocessing/build_dataset.py --exp exp5 --start_date 19800101 --end_date 19800101
