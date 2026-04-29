#!/bin/bash
#SBATCH --job-name=exp5_train
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

module load python/gloenv3.12_arm
module load nvidia/cuda/12.4

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export IDOWNSCALE_RAW_DIR="${IDOWNSCALE_RAW_DIR:-${REPO_ROOT}/rawdata}"
export IDOWNSCALE_OUTPUT_DIR="${IDOWNSCALE_OUTPUT_DIR:-/gpfs-calypso/scratch/globc/page/idownscale_output}"
export IDOWNSCALE_REGRID_WEIGHTS_DIR="${IDOWNSCALE_REGRID_WEIGHTS_DIR:-${IDOWNSCALE_OUTPUT_DIR}/weights}"
export IDOWNSCALE_VENV_PATH="${IDOWNSCALE_VENV_PATH:-}"
export IDOWNSCALE_EXTRA_PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH:-}"
export IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES="${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES:-}"

if [[ -n "${IDOWNSCALE_VENV_PATH}" ]]; then
  source "${IDOWNSCALE_VENV_PATH}/bin/activate"
  if [[ -n "${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES}" ]]; then
    python -m pip install ${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES}
  fi
fi

if [[ -n "${IDOWNSCALE_EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
fi

cd "${REPO_ROOT}"

TEST_NAME="${TEST_NAME:-unet_smoke}"
MAX_EPOCH="${MAX_EPOCH:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.0008}"
TRAIN_MODEL="${TRAIN_MODEL:-unet}"
IF_EXISTS="${IF_EXISTS:-skip}"
STEPS="${STEPS:-phase1,stats,train}"

echo "--- exp5 training smoke start: $(date) ---"
echo "python: $(command -v python)"
echo "IDOWNSCALE_VENV_PATH=${IDOWNSCALE_VENV_PATH:-<none>}"
echo "IDOWNSCALE_EXTRA_PYTHONPATH=${IDOWNSCALE_EXTRA_PYTHONPATH:-<none>}"
echo "IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES=${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES:-<none>}"
bash bin/production/run_exp5_workflow_grace.sh \
  --exp exp5 \
  --steps "${STEPS}" \
  --if-exists "${IF_EXISTS}" \
  --test-name "${TEST_NAME}" \
  --train-max-epoch "${MAX_EPOCH}" \
  --train-batch-size "${BATCH_SIZE}" \
  --train-learning-rate "${LEARNING_RATE}" \
  --train-model "${TRAIN_MODEL}"

echo "--- exp5 training smoke end: $(date) ---"
