#!/bin/bash
#SBATCH --job-name=exp5_eval
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

TEST_NAME="${TEST_NAME:-unet_grace30}"
SIMU_TEST="${SIMU_TEST:-gcm_bc}"
IF_EXISTS="${IF_EXISTS:-overwrite}"
PREDICT_START_DATE="${PREDICT_START_DATE:-}"
PREDICT_END_DATE="${PREDICT_END_DATE:-}"
METRICS_START_DATE="${METRICS_START_DATE:-}"
METRICS_END_DATE="${METRICS_END_DATE:-}"
VALUE_START_DATE="${VALUE_START_DATE:-}"
VALUE_END_DATE="${VALUE_END_DATE:-}"
STEPS="${STEPS:-predict_loop,metrics_day,metrics_month,value_metrics,plot_metrics_day,plot_metrics_month}"
CHECKPOINT_BUNDLE="${CHECKPOINT_BUNDLE:-}"

echo "--- exp5 inference/eval start: $(date) ---"
echo "python: $(command -v python)"
echo "python3: $(command -v python3)"
echo "IDOWNSCALE_VENV_PATH=${IDOWNSCALE_VENV_PATH:-<none>}"
echo "CHECKPOINT_BUNDLE=${CHECKPOINT_BUNDLE:-<none>}"

CMD=(
  python
  bin/production/run_obs_workflow.py
  --exp exp5
  --steps "${STEPS}"
  --if-exists "${IF_EXISTS}"
  --test-name "${TEST_NAME}"
  --simu-test "${SIMU_TEST}"
)

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

echo "command: ${CMD[*]}"
"${CMD[@]}"

echo "--- exp5 inference/eval end: $(date) ---"
