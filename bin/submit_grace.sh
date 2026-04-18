#!/bin/bash
#SBATCH --partition grace
#SBATCH --job-name idownscale_grace
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

# Ensure we are in the project root
ROOT_DIR="/gpfs-calypso/scratch/globc/page/idownscale_rerun"
cd "$ROOT_DIR" || exit 1

# Ensure slurm_logs directory exists
mkdir -p "$ROOT_DIR/slurm_logs"

echo "--- Job starting at $(date) ---"
echo "Project Root: $ROOT_DIR"
echo "Arguments: $@"

# 1. Verification
if [[ $(git status --porcelain) ]]; then
    echo "[GIT] Warning: Detected uncommitted changes. Please commit manualy if needed."
else
    echo "[GIT] Working directory is clean."
fi

# 2. Run the job using the grace wrapper
# Usage examples:
# sbatch bin/submit_grace.sh bin/training/train.py
# sbatch bin/submit_grace.sh bin/training/predict.py --date 20121018 --exp exp5
if [ $# -eq 0 ]; then
    echo "ERROR: No script provided to run."
    exit 1
fi

./bin/run_grace.sh "$@"

echo "--- Job finished at $(date) ---"
