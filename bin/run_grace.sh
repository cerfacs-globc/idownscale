#!/bin/bash

# Configuration
IMAGE="/softs/local_arm/singularity/images/pytorch25.02.sif"
VIRTUAL_ENV="/scratch/globc/page/idownscale_envs/env_idownscale_singularity"
PYTHON_BIN="${VIRTUAL_ENV}/bin/python3"

# Ensure output directory exists
export IDOWNSCALE_OUTPUT_DIR="/scratch/globc/page/idownscale_output"
mkdir -p "$IDOWNSCALE_OUTPUT_DIR"

# Check if image exists
if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Singularity image not found at $IMAGE"
    exit 1
fi

# Unset host python variables to avoid interference inside container
unset PYTHONHOME
unset PYTHONPATH

# Point to the fixed ESMF installation
ESMF_ROOT="/scratch/globc/page/idownscale_envs/esmf_fixed"
export PYTHONPATH="$ESMF_ROOT/lib/python3.12/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$ESMF_ROOT/lib:$LD_LIBRARY_PATH"
export ESMFMKFILE="$ESMF_ROOT/lib/esmf.mk"

# Run command inside singularity
# --nv: enable GPU support
# -B /scratch/: mount scratch filesystem
singularity run --nv -B /scratch/ "$IMAGE" "$PYTHON_BIN" "$@"
