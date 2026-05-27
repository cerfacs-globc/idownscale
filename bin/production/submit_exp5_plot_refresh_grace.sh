#!/bin/bash
# Refresh exp5 metric plots plus the historical 5-curve diagnostic on Grace.

#SBATCH --job-name=exp5_plot_refresh
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

module load python/gloenv3.12_arm

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export IDOWNSCALE_RAW_DIR="${IDOWNSCALE_RAW_DIR:-${REPO_ROOT}/rawdata}"
export IDOWNSCALE_OUTPUT_DIR="${IDOWNSCALE_OUTPUT_DIR:-/gpfs-calypso/scratch/globc/page/idownscale_output}"

if [[ -n "${IDOWNSCALE_EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
fi

cd "${REPO_ROOT}"

echo "--- Refresh exp5 metric plots: $(date) ---"
bash bin/production/run_exp5_workflow_grace.sh \
  --exp exp5 \
  --steps plot_metrics_day,plot_metrics_month \
  --test-name unet_all \
  --simu-test gcm_bc \
  --if-exists overwrite

echo "--- Build exp5 historical 5-curve PDF: $(date) ---"
python3 bin/evaluation/plot_exp5_historical_5curve.py

echo "--- Build exp5 pairwise distribution summary: $(date) ---"
python3 bin/evaluation/plot_exp5_pairwise_distribution_quantiles.py

echo "--- Plot refresh complete: $(date) ---"
