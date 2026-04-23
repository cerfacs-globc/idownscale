#!/bin/bash
# ------------------------------------------------------------------------------
#  run_egu_hands_on_v86.sh — EGU26 Machine-Portable Hands-on Master Script
#
#  This script orchestrates the Climate Downscaling Pipeline (Phases 1 & 2).
#  It is designed for portability across Laptops (macOS/WSL) and HPC environments.
#
#  Usage:
#    ./bin/run_egu_hands_on_v86.sh
# ------------------------------------------------------------------------------

# 1. Dynamic Root Discovery
REPO_ROOT=$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")
cd "$REPO_ROOT" || exit 1

# 2. Environment & Data Configuration
# Outputs go to the current directory unless IDOWNSCALE_DATA_DIR is set.
export IDOWNSCALE_DATA_DIR="${IDOWNSCALE_DATA_DIR:-$PWD}"
export IDOWNSCALE_RAW_DIR="${IDOWNSCALE_RAW_DIR:-$REPO_ROOT/rawdata}"

# Clean environment to avoid conflicting local packages
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# 3. Initialization
EXP="exp5"
log_file="$IDOWNSCALE_DATA_DIR/egu_workshop_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$log_file"
}

log "--- Initializing EGU26 Hands-on Workflow (v86.74) ---"
log "Repository Root: $REPO_ROOT"
log "Data Output: $IDOWNSCALE_DATA_DIR"

# 4. Phase 1: High-Resolution Reconstruction (ERA5)
log "--- Phase 1: RECONSTRUCTION START ---"
log "Creating training/validation samples..."
python3 bin/preprocessing/build_dataset.py --exp "$EXP" | tee -a "$log_file"

log "Computing training statistics..."
python3 bin/preprocessing/compute_statistics.py --exp "$EXP" | tee -a "$log_file"

# 5. Phase 2: Bias-Corrected Synthesis (GCM)
log "--- Phase 2: SYNTHESIS START ---"
log "Generating 120-year multi-period volumes (Historical & Future)..."
python3 bin/preprocessing/build_dataset_bc.py --simu gcm --ssp ssp585 --var tas | tee -a "$log_file"

log "--- WORKFLOW COMPLETE ---"
log "Results saved to $IDOWNSCALE_DATA_DIR/datasets/"
log "Log file: $log_file"
