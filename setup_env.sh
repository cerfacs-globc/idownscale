#!/bin/bash
#SBATCH -p grace
#SBATCH --gres=gpu:1
#SBATCH --job-name=setup_env
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=02:00:00

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUNTIME_ROOT="${IDOWNSCALE_RUNTIME_ROOT:-/scratch/globc/${USER}/idownscale_runtime}"
cd "$REPO_ROOT" || exit 1

# ── Scratch-local caches (keep everything off the home quota) ─────────────────
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${RUNTIME_ROOT}/conda/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUNTIME_ROOT}/pip_cache}"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

# ── Load conda (needed to create/install into the env) ───────────────────────
module load python/anaconda3.11_arm

CONDA_PREFIX="${IDOWNSCALE_CONDA_PREFIX:-${RUNTIME_ROOT}/conda/envs/idownscale_env}"

# ── Start from zero ────────────────────────────────────────────────────────────
echo "=== Removing old environment ==="
rm -rf "$CONDA_PREFIX"

# ── Create fresh conda environment ───────────────────────────────────────────
echo "=== Creating fresh environment ==="
# Pin python to conda-forge so the version string is already conda-forge style
# from the start (avoids mid-resolution Python downgrades).
conda create -y -p "$CONDA_PREFIX" -c conda-forge python=3.11

# ── Install all scientific packages via conda ─────────────────────────────────
echo "=== Installing conda packages ==="
conda install -y -p "$CONDA_PREFIX" -c conda-forge \
    eigen esmpy xesmf netcdf4 cartopy \
    numpy scipy pandas xarray \
    matplotlib seaborn pyproj \
    pytorch torchvision torchaudio pytorch-lightning torchmetrics \
    timm tqdm

# ── KEY WORKAROUND: unset PYTHONHOME before calling pip ───────────────────────
# `module load python/anaconda3.11_arm` sets PYTHONHOME to the system Anaconda.
# This causes the conda env's Python binary to load system site-packages
# (including a broken attr/_compat.py triggered by pip). Unsetting PYTHONHOME
# forces the conda env's Python to use only its own stdlib and site-packages.
echo "=== Isolating conda env from system Anaconda and ~/.local packages ==="
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1  # ignore ~/.local which may have stale packages

# ── Install pip-only packages into the conda env ─────────────────────────────
export ESMFMKFILE="$CONDA_PREFIX/lib/esmf.mk"
echo "=== Installing pip-only packages ==="
"$CONDA_PREFIX/bin/pip" install \
    --prefix "$CONDA_PREFIX" \
    --no-cache-dir \
    ibicus==1.1.1 \
    monai==1.4.0 \
    SBCK==1.4.2

# ── Install the project itself ────────────────────────────────────────────────
echo "=== Installing idownscale project ==="
"$CONDA_PREFIX/bin/pip" install \
    --prefix "$CONDA_PREFIX" \
    --no-cache-dir \
    -e .

# ── Verify ────────────────────────────────────────────────────────────────────
echo "=== Environment verification ==="
export ESMFMKFILE="$CONDA_PREFIX/lib/esmf.mk"
export PYTHONNOUSERSITE=1
"$CONDA_PREFIX/bin/python" -c "
import torch, xesmf, ibicus, cartopy
print('torch   :', torch.__version__, '| GPU:', torch.cuda.is_available())
print('xesmf   :', xesmf.__version__)
print('ibicus  :', ibicus.__version__)
print('cartopy :', cartopy.__version__)
print('ALL OK')
"
