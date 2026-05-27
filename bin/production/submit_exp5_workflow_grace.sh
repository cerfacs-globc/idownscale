#!/bin/bash
# Generic Calypso Grace submitter for exp5 workflow phases.

#SBATCH --job-name=exp5_workflow
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
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
export IDOWNSCALE_VENV_PATH="${IDOWNSCALE_VENV_PATH:-/scratch/globc/page/idownscale_envs/production_final_v22_312}"
export IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES="${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES:-}"
export IDOWNSCALE_EXTRA_PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH:-}"

if [[ -n "${IDOWNSCALE_VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${IDOWNSCALE_VENV_PATH}/bin/activate"
  if [[ -n "${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES}" ]]; then
    python -m pip install ${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES}
  fi
fi

if [[ -n "${IDOWNSCALE_EXTRA_PYTHONPATH}" ]]; then
  export PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
fi

cd "${REPO_ROOT}"

STEPS="${STEPS:-phase1,stats,bc_dataset,bc_apply}"
IF_EXISTS="${IF_EXISTS:-skip}"
PHASE1_START_DATE="${PHASE1_START_DATE:-}"
PHASE1_END_DATE="${PHASE1_END_DATE:-}"
SIMU="${SIMU:-gcm}"
VAR="${VAR:-tas}"
SSP="${SSP:-ssp585}"
TEST_NAME="${TEST_NAME:-}"
SIMU_TEST="${SIMU_TEST:-gcm_bc}"
PREDICT_START_DATE="${PREDICT_START_DATE:-20000101}"
PREDICT_END_DATE="${PREDICT_END_DATE:-21001231}"
CHECKPOINT_BUNDLE="${CHECKPOINT_BUNDLE:-}"
TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.0008}"
TRAIN_MODEL="${TRAIN_MODEL:-unet}"
TRAIN_LOSS="${TRAIN_LOSS:-}"

CMD=(
  bash
  bin/production/run_exp5_workflow_grace.sh
  --exp exp5
  --steps "${STEPS}"
  --if-exists "${IF_EXISTS}"
  --simu "${SIMU}"
  --var "${VAR}"
  --ssp "${SSP}"
  --simu-test "${SIMU_TEST}"
  --predict-start-date "${PREDICT_START_DATE}"
  --predict-end-date "${PREDICT_END_DATE}"
  --train-max-epoch "${TRAIN_MAX_EPOCH}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --train-learning-rate "${TRAIN_LEARNING_RATE}"
  --train-model "${TRAIN_MODEL}"
)

if [[ -n "${PHASE1_START_DATE}" ]]; then
  CMD+=(--phase1-start-date "${PHASE1_START_DATE}")
fi
if [[ -n "${PHASE1_END_DATE}" ]]; then
  CMD+=(--phase1-end-date "${PHASE1_END_DATE}")
fi
if [[ -n "${TEST_NAME}" ]]; then
  CMD+=(--test-name "${TEST_NAME}")
fi
if [[ -n "${CHECKPOINT_BUNDLE}" ]]; then
  CMD+=(--checkpoint-bundle "${CHECKPOINT_BUNDLE}")
fi
if [[ -n "${TRAIN_LOSS}" ]]; then
  CMD+=(--train-loss "${TRAIN_LOSS}")
fi

echo "--- exp5 Grace workflow start: $(date) ---"
echo "repo_root=${REPO_ROOT}"
echo "steps=${STEPS}"
echo "python=$(command -v python || true)"
echo "venv=${IDOWNSCALE_VENV_PATH:-<none>}"
echo "command: ${CMD[*]}"
"${CMD[@]}"
echo "--- exp5 Grace workflow end: $(date) ---"
