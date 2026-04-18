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

# Explicitly add Garcia's ESMF/esmpy build to the container's PYTHONPATH
ESMPY_PATH="/scratch/globc/page/idownscale_exp5/utils/esmf/src/addon/esmpy/build/lib"
export PYTHONPATH="$ESMPY_PATH"

# Run command inside singularity
# --nv: enable GPU support
# -B /scratch/: mount scratch filesystem
singularity run --nv -B /scratch/ "$IMAGE" "$PYTHON_BIN" "$@"
