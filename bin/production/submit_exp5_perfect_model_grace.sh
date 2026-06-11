#!/bin/bash
# Standalone Grace submitter for the RCM perfect-model workflow.

#SBATCH --job-name=perfect_model_rcm
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
SCRATCH_ROOT_DEFAULT="${SCRATCH_ROOT:-/scratch/globc/${USER}}"
RUNTIME_ROOT_DEFAULT="${SCRATCH_ROOT_DEFAULT}/idownscale_runtime"
RAW_ROOT_DEFAULT="${SCRATCH_ROOT_DEFAULT}/idownscale_rawdata"

module load python/gloenv3.12_arm
module load nvidia/cuda/12.4

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
export IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES="${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES:-}"
export IDOWNSCALE_EXTRA_PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH:-}"
export IDOWNSCALE_FORCE_VENV_SITEPACKAGES="${IDOWNSCALE_FORCE_VENV_SITEPACKAGES:-}"
export ESMFMKFILE="${ESMFMKFILE:-/softs/local_arm/Anaconda/2024.02-1/envs/gloenv_py3.11_arm/lib/esmf.mk}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -n "${IDOWNSCALE_VENV_PATH}" ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ -n "${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES}" ]]; then
    "${PYTHON_BIN}" -m pip install ${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES}
  fi
fi

if [[ -n "${IDOWNSCALE_FORCE_VENV_SITEPACKAGES}" ]]; then
  export PYTHONPATH="${IDOWNSCALE_FORCE_VENV_SITEPACKAGES}:${PYTHONPATH:-}"
fi

if [[ -n "${IDOWNSCALE_EXTRA_PYTHONPATH}" ]]; then
  export PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
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

echo "--- RCM perfect-model Grace workflow start: $(date) ---"
echo "repo_root=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "venv=${IDOWNSCALE_VENV_PATH:-<none>}"
echo "command: ${CMD[*]}"
"${CMD[@]}"
echo "--- RCM perfect-model Grace workflow end: $(date) ---"
