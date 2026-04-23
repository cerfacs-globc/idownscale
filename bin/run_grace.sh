#!/bin/bash

# Configuration
# native ARM python runner based on user gloenv guidance
# No singularity for preprocessing.

# AGGRESSIVELY unset Anaconda/x86 variables to avoid 'encodings' mismatch
# and ensure we pick up the correct system modules.
unset PYTHONHOME
unset PYTHONPATH
export PYTHONHOME=
export PYTHONPATH=

# Load the ARM-native python environment
module load python/gloenv3.12_arm

# Ensure base output directories exist (canonical paths)
export IDOWNSCALE_OUTPUT_DIR="/gpfs-calypso/scratch/globc/page/idownscale_output"
mkdir -p "$IDOWNSCALE_OUTPUT_DIR/datasets"
mkdir -p "$IDOWNSCALE_OUTPUT_DIR/runs"
mkdir -p "$IDOWNSCALE_OUTPUT_DIR/graph"
mkdir -p "$IDOWNSCALE_OUTPUT_DIR/metrics"
mkdir -p "$IDOWNSCALE_OUTPUT_DIR/prediction"

# Execute command explicitly with python3
python3 -u "$@"
