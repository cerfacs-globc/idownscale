#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=setup_env_%j.out
#SBATCH --error=setup_env_%j.err
#SBATCH --time=02:00:00

# ==============================================================================
#  setup_env.sh — Portable environment setup for idownscale
#
#  Builds a Python 3.11 venv at $IDOWNSCALE_ENVS/env_idownscale_arm using
#  the ARM-native Anaconda Python interpreter found via `module load`.
#
#  Portability:
#    - No hardcoded hostnames (works on any cluster).
#    - SCRATCH is auto-detected from the repo location if not set.
#    - IDOWNSCALE_ENVS defaults to $SCRATCH/../idownscale_envs (sibling of repo).
#
#  Usage:
#    sbatch -p <partition> --gres=gpu:1 setup_env.sh
#    # or interactively inside a salloc session:
#    bash setup_env.sh
# ==============================================================================

set -e

cd "$(dirname "$(realpath "$0")")" || exit 1
REPO_ROOT=$(pwd)

# ── Derive paths portably ─────────────────────────────────────────────────────
# Auto-detect scratch root from the repo path (parent of the repo dir).
SCRATCH="${SCRATCH:-$(dirname "$REPO_ROOT")}"
IDOWNSCALE_ENVS="${IDOWNSCALE_ENVS:-$SCRATCH/../idownscale_envs}"
IDOWNSCALE_ENVS=$(realpath -m "$IDOWNSCALE_ENVS")

VENV="$IDOWNSCALE_ENVS/env_idownscale_arm"
CACHE_DIR="$SCRATCH/../.cache"

echo "=== Setup ==="
echo "  REPO_ROOT : $REPO_ROOT"
echo "  VENV      : $VENV"
echo "  CACHE_DIR : $CACHE_DIR"

# ── Caches off home quota ─────────────────────────────────────────────────────
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$PIP_CACHE_DIR" "$IDOWNSCALE_ENVS"

# ── Load ARM python (provides python 3.11 at /softs/local_arm/...) ────────────
# We only use `module load` to discover the interpreter path; we immediately
# unset PYTHONHOME/PYTHONPATH so the venv is isolated.
module load python/anaconda3.11_arm
ARM_PYTHON=$(which python3)
echo "=== ARM Python interpreter: $ARM_PYTHON ($(python3 --version)) ==="
unset PYTHONHOME
unset PYTHONPATH

# ── Create fresh venv ─────────────────────────────────────────────────────────
echo "=== Creating venv ==="
rm -rf "$VENV"
env -u PYTHONHOME -u PYTHONPATH "$ARM_PYTHON" -m venv "$VENV"

PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

echo "=== Upgrading pip ==="
env -u PYTHONHOME -u PYTHONPATH "$PIP" install --upgrade pip

# ── Install all dependencies ──────────────────────────────────────────────────
# Install xesmf (needs ESMF) via conda-forge first via pip wheel
echo "=== Installing scientific stack ==="
env -u PYTHONHOME -u PYTHONPATH "$PIP" install \
    xarray dask netCDF4 bottleneck h5netcdf \
    scipy numpy pandas pyproj \
    matplotlib seaborn cartopy \
    xesmf

echo "=== Installing ML stack ==="
env -u PYTHONHOME -u PYTHONPATH "$PIP" install \
    torch torchvision pytorch_lightning torchmetrics timm monai tqdm

echo "=== Installing climate correction stack ==="
env -u PYTHONHOME -u PYTHONPATH "$PIP" install \
    ibicus SBCK

echo "=== Installing project (editable) ==="
env -u PYTHONHOME -u PYTHONPATH "$PIP" install --no-cache-dir -e "$REPO_ROOT"

# ── Verify ────────────────────────────────────────────────────────────────────
echo "=== Verification ==="
env -u PYTHONHOME -u PYTHONPATH "$PYTHON" -c "
import torch, xarray, ibicus, cartopy, SBCK
print('torch   :', torch.__version__, '| GPU:', torch.cuda.is_available())
print('xarray  :', xarray.__version__)
print('ibicus  :', ibicus.__version__)
print('SBCK    :', SBCK.__version__)
print('cartopy :', cartopy.__version__)
print('ALL OK')
"

echo ""
echo "=== Environment ready at: $VENV ==="
echo "=== To use: export PYTHON=$VENV/bin/python ==="
