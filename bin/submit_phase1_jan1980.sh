#!/bin/bash
#SBATCH --job-name=p1_jan1980
#SBATCH --output=slurm_logs/p1_jan1980_%j.out
#SBATCH --error=slurm_logs/p1_jan1980_%j.err
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

cd /gpfs-calypso/scratch/globc/page/idownscale_rerun || exit 1
./bin/run_grace.sh python3 -u bin/preprocessing/build_dataset.py \
    --exp exp5 \
    --start_date 1980-01-01 \
    --end_date 1980-01-31 \
    --audit_dir /scratch/globc/page/idownscale_output/audit_month/p1
