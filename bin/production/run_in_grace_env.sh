#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

module load python/gloenv3.12_arm

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export IDOWNSCALE_RAW_DIR="${IDOWNSCALE_RAW_DIR:-${REPO_ROOT}/rawdata}"
export IDOWNSCALE_OUTPUT_DIR="${IDOWNSCALE_OUTPUT_DIR:-/gpfs-calypso/scratch/globc/page/idownscale_output}"
export IDOWNSCALE_REGRID_WEIGHTS_DIR="${IDOWNSCALE_REGRID_WEIGHTS_DIR:-${IDOWNSCALE_OUTPUT_DIR}/weights}"
export IDOWNSCALE_VENV_PATH="${IDOWNSCALE_VENV_PATH:-/scratch/globc/page/idownscale_envs/production_final_v22_312}"
export IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES="${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES:-}"
export IDOWNSCALE_EXTRA_PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH:-}"
export IDOWNSCALE_FORCE_VENV_SITEPACKAGES="${IDOWNSCALE_FORCE_VENV_SITEPACKAGES:-}"

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
exec "$@"
