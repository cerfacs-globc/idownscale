#!/bin/bash
#SBATCH -p grace
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp5_full
#SBATCH --output=/scratch/globc/page/idownscale_active/exp5_full_%j.out
#SBATCH --error=/scratch/globc/page/idownscale_active/exp5_full_%j.err
#SBATCH --time=12:00:00

# --- Configuration ---
# You can override these via the environment if needed
EXP=${EXP:-exp5}
VAR=${VAR:-tas}
SSP=${SSP:-ssp585}
SIMU=${SIMU:-gcm}
TEST_NAME=${TEST_NAME:-unet_all}
SIMU_TEST=${SIMU_TEST:-gcm_bc}
START_DATE_INF=${START_DATE_INF:-20150101}
END_DATE_INF=${END_DATE_INF:-21001231}
# ---------------------

set -e
cd /scratch/globc/page/idownscale_active || exit 1

module load python/anaconda3.11_arm

CONDA_PREFIX="/scratch/globc/page/conda/envs/idownscale_env"
PYTHON="$CONDA_PREFIX/bin/python"

# Isolate from system Anaconda (PYTHONHOME) and stale ~/.local packages
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Required for xesmf
export ESMFMKFILE="$CONDA_PREFIX/lib/esmf.mk"
export PYTHONUNBUFFERED=1

# --- Modular execution control ---
# Use START_PHASE and STOP_PHASE to skip ranges (e.g. START_PHASE=4 ./run_exp5_full.sh)
# Use FORCE=1 to ignore existing .done markers and run the step.
# Use REGENERATE=1 to force regeneration from 0 (bypass resumability within scripts).
START_PHASE=${START_PHASE:-1}
STOP_PHASE=${STOP_PHASE:-6}
FORCE=${FORCE:-0}
REGENERATE=${REGENERATE:-0}
MARKER_DIR=".markers"
PROGRESS_LOG="PROGRESS.log"
mkdir -p "$MARKER_DIR"

FORCE_FLAG=""
if [[ $REGENERATE -eq 1 || $FORCE -eq 1 ]]; then
    FORCE_FLAG="--force"
    [[ $REGENERATE -eq 1 ]] && FORCE=1  # Ensure markers are bypassed if regenerating
fi

log_progress() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PROGRESS_LOG"
}

run_phase() {
    local phase_num=$1
    local marker="$MARKER_DIR/phase$phase_num.done"

    if [[ $phase_num -lt $START_PHASE || $phase_num -gt $STOP_PHASE ]]; then
        return 1 # Skip due to range
    fi

    if [[ -f "$marker" && $FORCE -ne 1 ]]; then
        log_progress "Phase $phase_num: SKIPPING (marker found)"
        return 1 # Skip due to marker
    fi

    return 0 # Should run
}

complete_phase() {
    local phase_num=$1
    touch "$MARKER_DIR/phase$phase_num.done"
    log_progress "Phase $phase_num: COMPLETED"
}

if run_phase 1; then
    log_progress "--- Phase 1: Preprocessing START ---"
    $PYTHON bin/preprocessing/build_dataset.py --exp "$EXP" $FORCE_FLAG
    $PYTHON bin/preprocessing/build_dataset.py --exp "$EXP" --baseline $FORCE_FLAG
    $PYTHON bin/preprocessing/compute_statistics.py --exp "$EXP" $FORCE_FLAG
    $PYTHON bin/preprocessing/compute_statistics_gamma.py --exp "$EXP" $FORCE_FLAG
    complete_phase 1
fi

if run_phase 2; then
    log_progress "--- Phase 2: Bias Correction Preprocessing START ---"
    $PYTHON bin/preprocessing/build_dataset_bc.py --simu "$SIMU" --ssp "$SSP" --var "$VAR" $FORCE_FLAG
    complete_phase 2
fi

if run_phase 3; then
    log_progress "--- Phase 3: Bias Correction (Ibicus) START ---"
    $PYTHON bin/preprocessing/bias_correction_ibicus.py --exp "$EXP" --ssp "$SSP" --simu "$SIMU" --var "$VAR" $FORCE_FLAG
    # After bias correction, we MUST compute statistics for the new dataset before training
    # Note: Using the naming convention datasets/dataset_bc/dataset_EXP_test_SIMU_bc
    BC_DATASET_DIR="datasets/dataset_bc/dataset_${EXP}_test_${SIMU}_bc"
    $PYTHON bin/preprocessing/compute_statistics.py --exp "$EXP" --dataset_dir "$BC_DATASET_DIR" $FORCE_FLAG
    $PYTHON bin/preprocessing/compute_statistics_gamma.py --exp "$EXP" --dataset_dir "$BC_DATASET_DIR" $FORCE_FLAG
    complete_phase 3
fi

if run_phase 4; then
    log_progress "--- Phase 4: Training START ---"
    srun $PYTHON bin/training/train.py
    complete_phase 4
fi

if run_phase 5; then
    log_progress "--- Phase 5: Inference START ---"
    srun $PYTHON bin/training/predict_loop.py --startdate "$START_DATE_INF" --enddate "$END_DATE_INF" --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" $FORCE_FLAG
    complete_phase 5
fi

if run_phase 6; then
    log_progress "--- Phase 6: Evaluation START ---"
    
    # 6.1 Future Trend Analysis (Qualitative)
    srun $PYTHON bin/evaluation/evaluate_futur_trend.py --exp "$EXP" --ssp "$SSP" --simu "$SIMU" $FORCE_FLAG
    
    # 6.2 Historical Validation (Quantitative - VALUE Framework)
    # First, run inference on the historical test period to allow comparison with ERA5
    log_progress "--- Phase 6.2: Historical Validation (VALUE) ---"
    srun $PYTHON bin/training/predict_loop.py --startdate 20000101 --enddate 20141231 --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" $FORCE_FLAG
    
    # Second, compute VALUE metrics for the historical period
    srun $PYTHON bin/evaluation/compute_value_metrics.py --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST"
    
    # 6.3 Automated PDF Reporting
    log_progress "--- Phase 6.3: Generating PDF Evaluation Report ---"
    srun $PYTHON bin/evaluation/generate_report.py --exp "$EXP" --test-name "$TEST_NAME" --simu "$SIMU" --ssp "$SSP"
    
    complete_phase 6
fi

log_progress "=== Workflow Sequence COMPLETE ==="
