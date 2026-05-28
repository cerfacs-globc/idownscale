#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

fail_with_layout_help() {
  echo "ERROR: ${1}" >&2
  echo >&2
  echo "Expected raw-data root: ${IDOWNSCALE_RAW_DIR}" >&2
  echo "Expected output root : ${IDOWNSCALE_OUTPUT_DIR}" >&2
  echo >&2
  echo "Recommended fixes:" >&2
  echo "  1. Put the raw files under ${REPO_ROOT}/rawdata" >&2
  echo "  2. or export IDOWNSCALE_RAW_DIR=/path/to/rawdata" >&2
  echo "  3. or create a symlink: ln -s /path/to/rawdata ${REPO_ROOT}/rawdata" >&2
  echo >&2
  echo "See doc/CALYPSO_RUNBOOK.md for the Calypso layout and commands." >&2
  exit 1
}

module load python/gloenv3.12_arm

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export IDOWNSCALE_RAW_DIR="${IDOWNSCALE_RAW_DIR:-${REPO_ROOT}/rawdata}"
export IDOWNSCALE_OUTPUT_DIR="${IDOWNSCALE_OUTPUT_DIR:-/gpfs-calypso/scratch/globc/page/idownscale_output}"
export IDOWNSCALE_REGRID_WEIGHTS_DIR="${IDOWNSCALE_REGRID_WEIGHTS_DIR:-${IDOWNSCALE_OUTPUT_DIR}/weights}"
export IDOWNSCALE_VENV_PATH="${IDOWNSCALE_VENV_PATH:-}"
export IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES="${IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES:-}"
export IDOWNSCALE_FORCE_VENV_SITEPACKAGES="${IDOWNSCALE_FORCE_VENV_SITEPACKAGES:-}"

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
python3 bin/production/run_exp5_workflow.py "$@"
