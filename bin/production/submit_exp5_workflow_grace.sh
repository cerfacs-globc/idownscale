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

if [[ -n "${IDOWNSCALE_VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${IDOWNSCALE_VENV_PATH}/bin/activate"
  if [[ -n "${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES}" ]]; then
    python -m pip install ${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES}
  fi
fi

if [[ -n "${IDOWNSCALE_FORCE_VENV_SITEPACKAGES}" ]]; then
  export PYTHONPATH="${IDOWNSCALE_FORCE_VENV_SITEPACKAGES}:${PYTHONPATH:-}"
fi

if [[ -n "${IDOWNSCALE_EXTRA_PYTHONPATH}" ]]; then
  export PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
fi

cd "${REPO_ROOT}"

STEPS="${STEPS:-phase1,stats,bc_dataset,bc_apply}"
IF_EXISTS="${IF_EXISTS:-skip}"
EXP="${EXP:-exp5}"
PHASE1_START_DATE="${PHASE1_START_DATE:-}"
PHASE1_END_DATE="${PHASE1_END_DATE:-}"
SIMU="${SIMU:-gcm}"
VAR="${VAR:-tas}"
SSP="${SSP:-ssp585}"
TEST_NAME="${TEST_NAME:-}"
SIMU_TEST="${SIMU_TEST:-gcm_bc}"
PREDICT_START_DATE="${PREDICT_START_DATE:-}"
PREDICT_END_DATE="${PREDICT_END_DATE:-}"
METRICS_START_DATE="${METRICS_START_DATE:-}"
METRICS_END_DATE="${METRICS_END_DATE:-}"
VALUE_START_DATE="${VALUE_START_DATE:-}"
VALUE_END_DATE="${VALUE_END_DATE:-}"
CHECKPOINT_BUNDLE="${CHECKPOINT_BUNDLE:-}"
TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.0008}"
TRAIN_MODEL="${TRAIN_MODEL:-unet}"
TRAIN_LOSS="${TRAIN_LOSS:-}"
BC_METHOD="${BC_METHOD:-}"
SAMPLE_START_DATE="${SAMPLE_START_DATE:-}"
SAMPLE_END_DATE="${SAMPLE_END_DATE:-}"

CMD=(
  bash
  bin/production/run_exp5_workflow_grace.sh
  --exp "${EXP}"
  --steps "${STEPS}"
  --if-exists "${IF_EXISTS}"
  --simu "${SIMU}"
  --var "${VAR}"
  --ssp "${SSP}"
  --simu-test "${SIMU_TEST}"
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
if [[ -n "${PREDICT_START_DATE}" ]]; then
  CMD+=(--predict-start-date "${PREDICT_START_DATE}")
fi
if [[ -n "${PREDICT_END_DATE}" ]]; then
  CMD+=(--predict-end-date "${PREDICT_END_DATE}")
fi
if [[ -n "${METRICS_START_DATE}" ]]; then
  CMD+=(--metrics-start-date "${METRICS_START_DATE}")
fi
if [[ -n "${METRICS_END_DATE}" ]]; then
  CMD+=(--metrics-end-date "${METRICS_END_DATE}")
fi
if [[ -n "${VALUE_START_DATE}" ]]; then
  CMD+=(--value-start-date "${VALUE_START_DATE}")
fi
if [[ -n "${VALUE_END_DATE}" ]]; then
  CMD+=(--value-end-date "${VALUE_END_DATE}")
fi
if [[ -n "${TRAIN_LOSS}" ]]; then
  CMD+=(--train-loss "${TRAIN_LOSS}")
fi
if [[ -n "${SAMPLE_START_DATE}" ]]; then
  CMD+=(--sample-start-date "${SAMPLE_START_DATE}")
fi
if [[ -n "${SAMPLE_END_DATE}" ]]; then
  CMD+=(--sample-end-date "${SAMPLE_END_DATE}")
fi
if [[ -n "${BC_METHOD}" ]]; then
  CMD+=(--bc-method "${BC_METHOD}")
fi

echo "--- exp5 Grace workflow start: $(date) ---"
echo "repo_root=${REPO_ROOT}"
echo "steps=${STEPS}"
echo "python=$(command -v python || true)"
echo "venv=${IDOWNSCALE_VENV_PATH:-<none>}"
echo "command: ${CMD[*]}"
"${CMD[@]}"
echo "--- exp5 Grace workflow end: $(date) ---"
