#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRATCH_ROOT_DEFAULT="${SCRATCH_ROOT:-/scratch/globc/${USER}}"
RUNTIME_ROOT_DEFAULT="${SCRATCH_ROOT_DEFAULT}/idownscale_runtime"
RAW_ROOT_DEFAULT="${SCRATCH_ROOT_DEFAULT}/idownscale_rawdata"

fail_with_layout_help() {
  echo "ERROR: ${1}" >&2
  echo >&2
  echo "Expected raw-data root: ${IDOWNSCALE_RAW_DIR}" >&2
  echo "Expected output root : ${IDOWNSCALE_OUTPUT_DIR}" >&2
  echo >&2
  echo "Recommended fixes:" >&2
  echo "  1. export IDOWNSCALE_RUNTIME_ROOT=${RUNTIME_ROOT_DEFAULT}" >&2
  echo "  2. export IDOWNSCALE_RAW_DIR=/path/to/rawdata" >&2
  echo "  3. or create a symlink: ln -s /path/to/rawdata ${REPO_ROOT}/rawdata" >&2
  echo >&2
  echo "See doc/CALYPSO_RUNBOOK.md for the Calypso layout and commands." >&2
  exit 1
}

module load python/gloenv3.12_arm

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
export IDOWNSCALE_FORCE_VENV_SITEPACKAGES="${IDOWNSCALE_FORCE_VENV_SITEPACKAGES:-}"
export ESMFMKFILE="${ESMFMKFILE:-/softs/local_arm/Anaconda/2024.02-1/envs/gloenv_py3.11_arm/lib/esmf.mk}"

if [[ -n "${IDOWNSCALE_EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
fi

if [[ -n "${IDOWNSCALE_VENV_PATH}" ]]; then
  if [[ ! -x "${IDOWNSCALE_VENV_PATH}/bin/python" ]]; then
    fail_with_layout_help "IDOWNSCALE_VENV_PATH does not contain a usable python: ${IDOWNSCALE_VENV_PATH}"
  fi
  # shellcheck disable=SC1090
  source "${IDOWNSCALE_VENV_PATH}/bin/activate"
  if [[ -n "${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES}" ]]; then
    python -m pip install ${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES}
  fi
fi

if [[ -n "${IDOWNSCALE_FORCE_VENV_SITEPACKAGES}" ]]; then
  export PYTHONPATH="${IDOWNSCALE_FORCE_VENV_SITEPACKAGES}:${PYTHONPATH:-}"
fi

[[ -d "${IDOWNSCALE_RAW_DIR}" ]] || fail_with_layout_help "raw-data directory not found"
[[ -d "${IDOWNSCALE_RAW_DIR}/eobs" ]] || fail_with_layout_help "missing ${IDOWNSCALE_RAW_DIR}/eobs"
[[ -d "${IDOWNSCALE_RAW_DIR}/era5" ]] || fail_with_layout_help "missing ${IDOWNSCALE_RAW_DIR}/era5"
[[ -d "${IDOWNSCALE_RAW_DIR}/gcm" ]] || fail_with_layout_help "missing ${IDOWNSCALE_RAW_DIR}/gcm"

echo "repo_root=${REPO_ROOT}"
echo "raw_dir=${IDOWNSCALE_RAW_DIR}"
echo "output_dir=${IDOWNSCALE_OUTPUT_DIR}"
echo "python=$(command -v python3)"

cd "${REPO_ROOT}"
python3 bin/production/run_obs_workflow.py "$@"
