#!/bin/bash
#SBATCH --job-name=idownscale_production_v86
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out

# ------------------------------------------------------------------------------
#  idownscale_master_v86.sh - Grand Master Production Orchestrator
# ------------------------------------------------------------------------------
set -e

# 1. Environment Synchronization (Hardened v86.74)
unset PYTHONHOME
unset PYTHONPATH
module load python/gloenv3.12_arm
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages
export PYTHONNOUSERSITE=1

# ESMF Guard for ARM-native xESMF stability
VENV_PATH=$(python3 -c "import sys; import os; print(os.path.dirname(os.path.dirname(sys.executable)))")
export ESMFMKFILE="$VENV_PATH/lib/esmf.mk"

# 2. Global Parameters
EXP="exp5"
SSP="ssp585"
SIMU="gcm"
MARKER_DIR="/gpfs-calypso/scratch/globc/page/idownscale_output/.markers"
mkdir -p "$MARKER_DIR"

# 3. Execution Control Logic
run_phase() {
    if [[ -f "$MARKER_DIR/phase$1.done" ]]; then
        echo "--- Phase $1: SKIPPING (Marker found) ---"
        return 1
    fi
    return 0
}

complete_phase() {
    touch "$MARKER_DIR/phase$1.done"
    echo "--- Phase $1: COMPLETED ---"
}

echo "--- STARTING MASTER PRODUCTION WORKFLOW: $(date) ---"

# PHASE 2: Regional Bias Correction 
if run_phase 2; then
    echo "--- Phase 2: Bias Correction Synthesis (GCM-Native) ---"
    python3 bin/preprocessing/build_dataset_bc.py --simu "$SIMU" --exp "$EXP" --var tas --ssp "$SSP"
    complete_phase 2
fi

# PHASE 3.1: The Discretization Bridge & Adaptive Stats
if run_phase 3; then
    echo "--- Phase 3.1: Discretization & Adaptive Normalization ---"
    # Placeholder for parallel discretization (e.g. discretize_bc_parallel.py)
    # python3 bin/preprocessing/discretize_bc_parallel.py --exp "$EXP" --simu ${SIMU}_bc
    
    # CRITICAL: Re-compute stats for the Bias-Corrected simulated climate
    BC_DIR="/gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_${EXP}_test_${SIMU}_bc"
    python3 bin/preprocessing/compute_statistics.py --exp "$EXP" --dataset_dir "$BC_DIR"
    complete_phase 3
fi

# PHASE 5: High-Resolution Inference (Downscaling)
if run_phase 5; then
    echo "--- Phase 5: UNet Prediction Loop (Inference) ---"
    python3 bin/training/predict_loop.py --exp "$EXP" --test-name unet_all --simu-test "${SIMU}_bc" --startdate 19800101 --enddate 21001231
    complete_phase 5
fi

# PHASE 6: Scientific Evaluation (Impact Suite)
if run_phase 6; then
    echo "--- Phase 6: Generating EGU Metrics & Trend Plots ---"
    
    # 6.1 Quantitative metrics (VALUE Framework)
    python3 bin/evaluation/compute_value_metrics.py --exp "$EXP" --simu "$SIMU"
    
    # 6.2 Standard Performance Plots (Boxplots/Barplots)
    python3 bin/evaluation/plot_test_metrics.py --exp "$EXP" --scale daily
    
    # 6.3 Zoé's Archival Histograms
    python3 bin/evaluation/plot_histograms.py --exp "$EXP" --simu "$SIMU"
    
    # 6.4 Strategic Future Trends
    python3 bin/evaluation/evaluate_futur_trend.py --exp "$EXP" --ssp "$SSP" --simu "$SIMU"
    
    # 6.5 Signature 5-Curve Master PDF (France Domain)
    # Define paths for the comparison (Assuming historical period for validation)
    # For future scenario, swap historical paths accordingly
    RAW_PATH="data/GCM/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc"
    BC_PATH="/gpfs-calypso/scratch/globc/page/idownscale_output/datasets/dataset_bc/tas_day_CNRM-CM6-1_historical_r1i1p1f2_bc.nc"
    AI_PATH="/gpfs-calypso/scratch/globc/page/idownscale_output/predictions/${EXP}_unet_all_gcm_bc.nc"
    
    python3 bin/production/plot_scientific_master_pdf.py \
        --exp "$EXP" --ssp "$SSP" \
        --raw "$RAW_PATH" --bc "$BC_PATH" --ai "$AI_PATH"
    
    complete_phase 6
fi

echo "--- PRODUCTION WORKFLOW SUCCESSFUL: $(date) ---"
