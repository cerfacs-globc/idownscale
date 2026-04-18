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
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
else
    ROOT_DIR="$(cd "$(dirname "$0")/.." || exit 1; pwd)"
fi
cd "$ROOT_DIR" || exit 1

# Export PYTHONPATH for esmpy reference build
export PYTHONPATH="/scratch/globc/page/idownscale_exp5/utils/esmf/src/addon/esmpy/build/lib:$PYTHONPATH"

echo "--- Job starting at $(date) ---"
echo "Project Root: $ROOT_DIR"
echo "Arguments: $@"

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
