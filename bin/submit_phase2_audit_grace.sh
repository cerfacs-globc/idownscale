#!/bin/bash
#SBATCH --job-name=phase2_audit
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/phase2_audit_%j.out
#SBATCH --error=slurm_logs/phase2_audit_%j.err

# Leveraging the proven scientific foundations
bash bin/run_grace.sh python3 bin/verification/comprehensive_parity_audit_bc.py
