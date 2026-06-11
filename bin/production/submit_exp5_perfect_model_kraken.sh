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
SCRATCH_ROOT_DEFAULT="/scratch/globc/${USER}"
RUNTIME_ROOT_DEFAULT="${SCRATCH_ROOT_DEFAULT}/idownscale_runtime"
RAW_ROOT_DEFAULT="${SCRATCH_ROOT_DEFAULT}/idownscale_rawdata"

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export IDOWNSCALE_RUNTIME_ROOT="${IDOWNSCALE_RUNTIME_ROOT:-${RUNTIME_ROOT_DEFAULT}}"
if [[ -z "${IDOWNSCALE_RAW_DIR:-}" && -d "${REPO_ROOT}/rawdata" ]]; then
  export IDOWNSCALE_RAW_DIR="${REPO_ROOT}/rawdata"
else
  export IDOWNSCALE_RAW_DIR="${IDOWNSCALE_RAW_DIR:-${RAW_ROOT_DEFAULT}}"
fi
export IDOWNSCALE_OUTPUT_DIR="${IDOWNSCALE_OUTPUT_DIR:-${IDOWNSCALE_RUNTIME_ROOT}/output}"
export IDOWNSCALE_REGRID_WEIGHTS_DIR="${IDOWNSCALE_REGRID_WEIGHTS_DIR:-${IDOWNSCALE_OUTPUT_DIR}/regrid_weights}"
export IDOWNSCALE_VENV_PATH="${IDOWNSCALE_VENV_PATH:-}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

EXP="${EXP:-perfect_model_rcm}"
VAR="${VAR:-tas}"
SSP="${SSP:-ssp585}"
SIMU="${SIMU:-rcm}"
TEST_NAME="${TEST_NAME:-unet_perfect_model_rcm}"
STEPS="${STEPS:-all}"
STEPS="${STEPS//__/,}"
IF_EXISTS="${IF_EXISTS:-skip}"
TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.0008}"
TRAIN_MODEL="${TRAIN_MODEL:-unet}"
TRAIN_LOSS="${TRAIN_LOSS:-}"
TRAIN_OUTPUT_NORM="${TRAIN_OUTPUT_NORM:-0}"
TRAIN_SEED="${TRAIN_SEED:-}"
TRAIN_N_STEPS="${TRAIN_N_STEPS:-200}"
PREDICT_NUM_SAMPLES="${PREDICT_NUM_SAMPLES:-1}"
PERFECT_MODEL_TARGET_SOURCE="${PERFECT_MODEL_TARGET_SOURCE:-}"
VALIDATION_STARTDATE="${VALIDATION_STARTDATE:-}"
VALIDATION_ENDDATE="${VALIDATION_ENDDATE:-}"
VALIDATION_HISTORICAL_ENDDATE="${VALIDATION_HISTORICAL_ENDDATE:-}"
VALIDATION_UNIT="${VALIDATION_UNIT:-}"
WORK_STARTDATE="${WORK_STARTDATE:-}"
WORK_ENDDATE="${WORK_ENDDATE:-}"
SAMPLE_DIR="${SAMPLE_DIR:-}"
EVAL_SAMPLE_DIR="${EVAL_SAMPLE_DIR:-}"

CMD=(
  "${PYTHON_BIN}"
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

if [[ -n "${TRAIN_SEED}" ]]; then
  CMD+=(--train-seed "${TRAIN_SEED}")
fi

if [[ "${TRAIN_MODEL}" == "cddpm" ]]; then
  CMD+=(--train-n-steps "${TRAIN_N_STEPS}")
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

if [[ -n "${WORK_STARTDATE}" ]]; then
  CMD+=(--work-startdate "${WORK_STARTDATE}")
fi

if [[ -n "${WORK_ENDDATE}" ]]; then
  CMD+=(--work-enddate "${WORK_ENDDATE}")
fi

if [[ -n "${SAMPLE_DIR}" ]]; then
  CMD+=(--sample-dir "${SAMPLE_DIR}")
fi

if [[ -n "${EVAL_SAMPLE_DIR}" ]]; then
  CMD+=(--eval-sample-dir "${EVAL_SAMPLE_DIR}")
fi

if [[ -n "${PREDICT_NUM_SAMPLES}" ]]; then
  CMD+=(--predict-num-samples "${PREDICT_NUM_SAMPLES}")
fi

echo "--- RCM perfect-model Kraken workflow start: $(date) ---"
echo "repo_root=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "venv=${IDOWNSCALE_VENV_PATH:-<none>}"
echo "command: ${CMD[*]}"
"${CMD[@]}"
echo "--- RCM perfect-model Kraken workflow end: $(date) ---"
