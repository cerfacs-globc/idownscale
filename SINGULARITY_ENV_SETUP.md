# Singularity Environment Setup — Grace (ARM/AArch64) Nodes

This guide explains how to set up a fully isolated Python environment for **idownscale** inside the official NVIDIA PyTorch Singularity container on Calypso Grace nodes. This approach resolves CUDA/GPU linkage issues and avoids all x86/AMD library pollution.

---

## Architecture Overview

```
/softs/local_arm/singularity/images/pytorch25.02.sif   ← NVIDIA base image (ARM64)
           │
           ▼
/scratch/globc/page/idownscale_envs/
    ├── env_idownscale_singularity/   ← Main Python venv (pip-installed packages)
    ├── env_idownscale_<branch>/      ← Per-branch venvs (optional)
    └── esmf_fixed/                   ← Shared ESMF arm-native install (micromamba)
         ├── lib/libesmf_fullylinked.so
         ├── lib/esmf_container.mk    ← Path-sanitised .mk for inside the container
         └── lib/python3.12/site-packages/esmpy/
```

> [!IMPORTANT]
> The `esmf_fixed` directory is **shared** across all branch environments. It only needs to be built once (already done). The per-branch `env_*` directories are lightweight pip venvs.

---

## Prerequisites (one-time setup, already done)

### 1. Download SBCK source

```bash
mkdir -p /scratch/globc/page/tmp/sbck_build
cd /scratch/globc/page/tmp/sbck_build
pip download SBCK==1.4.2 --no-deps
tar -xzf SBCK-1.4.2.tar.gz      # or unzip, depending on format
```

### 2. Download Eigen 3 headers

```bash
mkdir -p /scratch/globc/page/tmp/eigen_download
cd /scratch/globc/page/tmp/eigen_download
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz
```

### 3. Build ESMF via micromamba (ARM-native)

This only needs to be done once and is shared by all environments.

```bash
# Download micromamba for aarch64
mkdir -p /scratch/globc/page/tmp
cd /scratch/globc/page/tmp
curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj bin/micromamba
mv bin/micromamba micromamba_arm

# Install esmf into the shared prefix
srun -p grace --gres=gpu:0 /scratch/globc/page/tmp/micromamba_arm install -y \
    -p /scratch/globc/page/idownscale_envs/esmf_fixed \
    -c conda-forge esmf esmpy

# Create a path-sanitised .mk file for use inside the container
# (The container sees /scratch/..., not /gpfs-calypso/scratch/...)
sed 's|/gpfs-calypso/scratch|/scratch|g' \
    /scratch/globc/page/idownscale_envs/esmf_fixed/lib/esmf.mk \
    > /scratch/globc/page/idownscale_envs/esmf_fixed/lib/esmf_container.mk
```

---

## Setting Up a Branch Environment

### Option A — Automated (recommended)

Use the SLURM script `setup_singularity_env.sh`. It accepts an optional environment name argument.

```bash
# Default name (env_idownscale_singularity)
sbatch setup_singularity_env.sh

# Branch-specific name
sbatch setup_singularity_env.sh env_idownscale_my_branch
```

The script will:
1. Create a fresh Python venv with `--system-site-packages` inside the container.
2. Install the full scientific and AI stack, including `SBCK` from source.
3. Link against the shared `esmf_fixed` installation.
4. Run a smoke test to confirm GPU visibility and all imports work.

---

### Option B — Manual (step-by-step)

Allocate a Grace node first:

```bash
srun -p grace --gres=gpu:1 --pty bash
```

All subsequent commands are run on the Grace node unless stated otherwise.

#### Step 1 — Create virtual environment

```bash
IMAGE="/softs/local_arm/singularity/images/pytorch25.02.sif"
ENV_DIR="/scratch/globc/page/idownscale_envs/env_idownscale_my_branch"

singularity exec --cleanenv --nv -B /scratch/ "$IMAGE" \
    python3 -m venv --system-site-packages "$ENV_DIR"
```

#### Step 2 — Install Python packages

```bash
singularity exec --cleanenv --nv -B /scratch/ "$IMAGE" bash -c "
    source ${ENV_DIR}/bin/activate
    export PYTHONNOUSERSITE=1

    pip install --upgrade pip setuptools wheel

    # Core scientific stack (numpy<2 required for numba compat with container)
    pip install 'numpy<2' 'scipy<1.14' pandas xarray dask

    # Geo / plotting
    pip install matplotlib seaborn tqdm cartopy pyproj

    # AI stack
    pip install pytorch-lightning==2.4.0 torchmetrics timm monai

    # Bias correction
    pip install ibicus==1.1.1

    # xesmf Python wrapper (the shared library comes from esmf_fixed)
    pip install xesmf
"
```

#### Step 3 — Build SBCK from source

SBCK requires Eigen 3 headers that are not included in the ARM wheels.

```bash
SBCK_SRC="/scratch/globc/page/tmp/sbck_build/sbck-1.4.2"
EIGEN_PATH="/scratch/globc/page/tmp/eigen_download/eigen-3.4.0"

singularity exec --cleanenv --nv -B /scratch/ "$IMAGE" bash -c "
    source ${ENV_DIR}/bin/activate
    export PYTHONNOUSERSITE=1
    export EIGEN_INCLUDE_PATH=\"${EIGEN_PATH}\"
    cd \"${SBCK_SRC}\"
    pip install .
"
```

#### Step 4 — Verify the environment

```bash
ESMF_PATH="/scratch/globc/page/idownscale_envs/esmf_fixed"

singularity run --cleanenv --nv -B /scratch/ "$IMAGE" bash -c "
    source ${ENV_DIR}/bin/activate
    export PYTHONNOUSERSITE=1
    export ESMFMKFILE=\"${ESMF_PATH}/lib/esmf_container.mk\"
    export LD_LIBRARY_PATH=\"${ESMF_PATH}/lib:\$LD_LIBRARY_PATH\"
    export PYTHONPATH=\"${ESMF_PATH}/lib/python3.12/site-packages:\$PYTHONPATH\"

    python -c \"
import torch, SBCK, esmpy, xesmf
print(f'torch {torch.__version__}  CUDA={torch.cuda.is_available()}')
print(f'SBCK OK  xesmf {xesmf.__version__}')
print('All imports successful.')
\"
"
```

---

## Running a Job

Use `run_singularity_template.sh` as a starting point. Copy and adapt it for your branch:

```bash
cp run_singularity_template.sh run_phases_5_7_my_branch.sh
# Edit ENV_NAME and the phase commands inside the file
sbatch run_phases_5_7_my_branch.sh
```

The key environment preamble used inside every `singularity run` call is:

```bash
ENV_PATH="/scratch/globc/page/idownscale_envs/env_idownscale_my_branch"
ESMF_PATH="/scratch/globc/page/idownscale_envs/esmf_fixed"

source ${ENV_PATH}/bin/activate
export PYTHONNOUSERSITE=1
export ESMFMKFILE="${ESMF_PATH}/lib/esmf_container.mk"
export LD_LIBRARY_PATH="${ESMF_PATH}/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="${ESMF_PATH}/lib/python3.12/site-packages:.:$PYTHONPATH"
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `--cleanenv` flag | Prevents AMD/x86 host paths from leaking into the container |
| `--system-site-packages` venv | Inherits PyTorch 2.7 + CUDA drivers from the NVIDIA base image |
| `numpy<2` pinned | Container's `numba` (0.59.1) requires NumPy ≤ 1.26 |
| ESMF via micromamba | `esmpy` has no ARM wheel on PyPI; micromamba installs a pre-compiled ARM-native binary |
| `esmf_container.mk` | The micromamba-installed ESMF has `/gpfs-calypso/scratch/...` hardcoded; the sanitised `.mk` maps correctly to `/scratch/...` inside the container |
| SBCK from source | No ARM binary wheel on PyPI; must be compiled with Eigen headers |

---

## Troubleshooting

### `ValueError: numpy.dtype size changed`
The container's system `scipy` is incompatible with a custom numpy. Ensure you have installed `numpy<2` inside the venv, **not** numpy 2.x.

```bash
# Inside the container
pip install 'numpy<2' 'scipy<1.14' --force-reinstall
```

### `ImportError: The esmf.mk file cannot be found`
Set `ESMFMKFILE` to the sanitised path:
```bash
export ESMFMKFILE="/scratch/globc/page/idownscale_envs/esmf_fixed/lib/esmf_container.mk"
```

### `OSError: libesmf_fullylinked.so: No such file or directory`
The container path differs from the host path. Make sure you are using `esmf_container.mk` (not `esmf.mk`) and that `LD_LIBRARY_PATH` includes the esmf_fixed lib directory.

### `torch.cuda.is_available()` returns `False`
Ensure you pass `--nv` to `singularity run` and request a GPU in SLURM (`--gres=gpu:1`).

### `ModuleNotFoundError` for a missing package
The venv may not have the full requirements. Reinstall inside the container:
```bash
singularity exec --cleanenv --nv -B /scratch/ "$IMAGE" bash -c "
    source ${ENV_DIR}/bin/activate
    pip install <missing-package>
"
```
