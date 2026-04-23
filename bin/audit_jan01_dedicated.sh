#!/bin/bash
#SBATCH --job-name=AUDIT_JAN01
#SBATCH --output=slurm_logs/audit_jan01_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --partition=grace
#SBATCH --time=00:05:00

./bin/run_grace.sh /scratch/globc/page/idownscale_rerun/scratch/verify_jan01_authenticated.py
