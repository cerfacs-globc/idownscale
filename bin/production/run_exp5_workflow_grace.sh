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

if [[ -n "${IDOWNSCALE_EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${IDOWNSCALE_EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
fi

cd "${REPO_ROOT}"
python3 bin/production/run_exp5_workflow.py "$@"
