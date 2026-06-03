#!/bin/bash
# Standalone Kraken submitter for the RCM perfect-model workflow.

#SBATCH --job-name=perfect_model_rcm
#SBATCH --partition=gpua30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export IDOWNSCALE_RAW_DIR="${IDOWNSCALE_RAW_DIR:-${REPO_ROOT}/rawdata}"
export IDOWNSCALE_OUTPUT_DIR="${IDOWNSCALE_OUTPUT_DIR:-/scratch/globc/page/idownscale_output}"
export IDOWNSCALE_REGRID_WEIGHTS_DIR="${IDOWNSCALE_REGRID_WEIGHTS_DIR:-${IDOWNSCALE_OUTPUT_DIR}/weights}"
export IDOWNSCALE_VENV_PATH="${IDOWNSCALE_VENV_PATH:-/scratch/globc/page/idownscale_envs/kraken_gpu_py312_v1}"

if [[ -n "${IDOWNSCALE_VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${IDOWNSCALE_VENV_PATH}/bin/activate"
fi

cd "${REPO_ROOT}"

EXP="${EXP:-perfect_model_rcm}"
VAR="${VAR:-tas}"
SSP="${SSP:-ssp585}"
SIMU="${SIMU:-rcm}"
TEST_NAME="${TEST_NAME:-unet_perfect_model_rcm}"
STEPS="${STEPS:-all}"
IF_EXISTS="${IF_EXISTS:-skip}"
TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.0008}"
TRAIN_MODEL="${TRAIN_MODEL:-unet}"
TRAIN_LOSS="${TRAIN_LOSS:-}"
TRAIN_OUTPUT_NORM="${TRAIN_OUTPUT_NORM:-0}"
PERFECT_MODEL_TARGET_SOURCE="${PERFECT_MODEL_TARGET_SOURCE:-}"
VALIDATION_STARTDATE="${VALIDATION_STARTDATE:-}"
VALIDATION_ENDDATE="${VALIDATION_ENDDATE:-}"
VALIDATION_HISTORICAL_ENDDATE="${VALIDATION_HISTORICAL_ENDDATE:-}"
VALIDATION_UNIT="${VALIDATION_UNIT:-}"

CMD=(
  python
  bin/production/run_exp5_perfect_model.py
  --exp "${EXP}"
  --var "${VAR}"
  --ssp "${SSP}"
  --simu "${SIMU}"
  --test-name "${TEST_NAME}"
  --steps "${STEPS}"
  --if-exists "${IF_EXISTS}"
  --train-max-epoch "${TRAIN_MAX_EPOCH}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --train-learning-rate "${TRAIN_LEARNING_RATE}"
  --train-model "${TRAIN_MODEL}"
)

if [[ -n "${TRAIN_LOSS}" ]]; then
  CMD+=(--train-loss "${TRAIN_LOSS}")
fi

if [[ "${TRAIN_OUTPUT_NORM}" == "1" ]]; then
  CMD+=(--train-output-norm)
fi

if [[ -n "${PERFECT_MODEL_TARGET_SOURCE}" ]]; then
  CMD+=(--perfect-model-target-source "${PERFECT_MODEL_TARGET_SOURCE}")
fi

if [[ -n "${VALIDATION_STARTDATE}" ]]; then
  CMD+=(--validation-startdate "${VALIDATION_STARTDATE}")
fi

if [[ -n "${VALIDATION_ENDDATE}" ]]; then
  CMD+=(--validation-enddate "${VALIDATION_ENDDATE}")
fi

if [[ -n "${VALIDATION_HISTORICAL_ENDDATE}" ]]; then
  CMD+=(--validation-historical-enddate "${VALIDATION_HISTORICAL_ENDDATE}")
fi

if [[ -n "${VALIDATION_UNIT}" ]]; then
  CMD+=(--validation-unit "${VALIDATION_UNIT}")
fi

echo "--- RCM perfect-model Kraken workflow start: $(date) ---"
echo "repo_root=${REPO_ROOT}"
echo "python=$(command -v python || true)"
echo "venv=${IDOWNSCALE_VENV_PATH:-<none>}"
echo "command: ${CMD[*]}"
"${CMD[@]}"
echo "--- RCM perfect-model Kraken workflow end: $(date) ---"
