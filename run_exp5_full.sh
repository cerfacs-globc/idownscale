#!/bin/bash
#SBATCH -p grace
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp5_full
#SBATCH --output=/scratch/globc/page/idownscale_active/logs/exp5_full_%j.out
#SBATCH --error=/scratch/globc/page/idownscale_active/logs/exp5_full_%j.err
#SBATCH --time=12:00:00

# ---------------------
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/$EXP/$RUN_ID"
export LOG_DIR
mkdir -p "$LOG_DIR"
# Ensure essential output and data directories exist for first-time runs
mkdir -p output/"$EXP"/validation output/"$EXP"/predictions
mkdir -p metrics/"$EXP" graph/metrics/"$EXP"
mkdir -p datasets/dataset_bc

set -e
cd /scratch/globc/page/idownscale_active || exit 1

module load python/anaconda3.11_arm

CONDA_PREFIX="/home/globc/page/.conda/envs/idownscale_env"
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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/execution.log"
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
    $PYTHON bin/preprocessing/build_dataset.py --exp "$EXP" $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    $PYTHON bin/preprocessing/build_dataset.py --exp "$EXP" --baseline $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    $PYTHON bin/preprocessing/compute_statistics.py --exp "$EXP" $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    $PYTHON bin/preprocessing/compute_statistics_gamma.py --exp "$EXP" $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    srun $PYTHON bin/utils/check_pipeline_integrity.py --phase 1 --exp "$EXP" | tee -a "output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 1
fi

if run_phase 2; then
    log_progress "--- Phase 2: Bias Correction Preprocessing START ---"
    $PYTHON bin/preprocessing/build_dataset_bc.py --simu "$SIMU" --ssp "$SSP" --var "$VAR" $FORCE_FLAG >> "$LOG_DIR/phase2.log" 2>&1
    srun $PYTHON bin/utils/check_pipeline_integrity.py --phase 2 --exp "$EXP" | tee -a "output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 2
fi

if run_phase 3; then
    log_progress "--- Phase 3: Bias Correction (Ibicus) START ---"
    $PYTHON bin/preprocessing/bias_correction_ibicus.py --exp "$EXP" --ssp "$SSP" --simu "$SIMU" --var "$VAR" $FORCE_FLAG >> "$LOG_DIR/phase3.log" 2>&1
    BC_DATASET_DIR="datasets/dataset_bc/dataset_${EXP}_test_${SIMU}_bc"
    $PYTHON bin/preprocessing/compute_statistics.py --exp "$EXP" --dataset_dir "$BC_DATASET_DIR" $FORCE_FLAG >> "$LOG_DIR/phase3.log" 2>&1
    $PYTHON bin/preprocessing/compute_statistics_gamma.py --exp "$EXP" --dataset_dir "$BC_DATASET_DIR" $FORCE_FLAG >> "$LOG_DIR/phase3.log" 2>&1
    srun $PYTHON bin/utils/check_pipeline_integrity.py --phase 3 --exp "$EXP" --simu "$SIMU" | tee -a "output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 3
fi

if run_phase 4; then
    log_progress "--- Phase 4: Training START ---"
    srun $PYTHON bin/training/train.py >> "$LOG_DIR/phase4.log" 2>&1
    srun $PYTHON bin/utils/check_pipeline_integrity.py --phase 4 --exp "$EXP" | tee -a "output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 4
fi

if run_phase 5; then
    log_progress "--- Phase 5: Inference START ---"
    srun $PYTHON bin/training/predict_loop.py --startdate "$START_DATE_INF" --enddate "$END_DATE_INF" --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" $FORCE_FLAG >> "$LOG_DIR/phase5.log" 2>&1
    srun $PYTHON bin/utils/check_pipeline_integrity.py --phase 5 --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" | tee -a "output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 5
fi

if run_phase 6; then
    log_progress "--- Phase 6: Evaluation START ---"
    
    # 6.0 Original Master Evaluation (Legacy - Optimized)
    log_progress "--- Phase 6.0: Original Master Evaluation (Legacy) ---"
    mkdir -p "graph/metrics/$EXP/$TEST_NAME_$SIMU_TEST"
    srun $PYTHON bin/evaluation/compute_test_metrics_day_fast.py --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" >> "$LOG_DIR/phase6_metrics.log" 2>&1
    srun $PYTHON bin/evaluation/plot_test_metrics.py --exp "$EXP" --test-name "${TEST_NAME}_${SIMU_TEST}" --scale daily >> "$LOG_DIR/phase6_plots.log" 2>&1
    
    # 6.1 Future Trend Analysis (Qualitative)
    srun $PYTHON bin/evaluation/evaluate_futur_trend.py --exp "$EXP" --ssp "$SSP" --simu "$SIMU" $FORCE_FLAG >> "$LOG_DIR/phase6_trends.log" 2>&1
    
    # 6.2 Historical Validation (Quantitative - VALUE Framework)
    log_progress "--- Phase 6.2: Historical Validation (VALUE) ---"
    srun $PYTHON bin/training/predict_loop.py --startdate 20000101 --enddate 20141231 --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" $FORCE_FLAG >> "$LOG_DIR/phase6_historical.log" 2>&1
    srun $PYTHON bin/evaluation/compute_value_metrics.py --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" >> "$LOG_DIR/phase6_value.log" 2>&1
    
    # 6.3 Automated PDF Reporting
    log_progress "--- Phase 6.3: Generating PDF Evaluation Report ---"
    srun $PYTHON bin/evaluation/generate_report.py --exp "$EXP" --test-name "$TEST_NAME" --simu "$SIMU" --ssp "$SSP" >> "$LOG_DIR/phase6_report.log" 2>&1
    
    # 6.4 Consolidate plots in output directory
    log_progress "--- Phase 6.4: Consolidating plots in output directory ---"
    VALIDATION_DIR="/scratch/globc/page/idownscale_active/output/$EXP/validation"
    mkdir -p "$VALIDATION_DIR"
    
    # Copy new SOTA plots
    cp graph/metrics/"$EXP"/*.png "$VALIDATION_DIR/" 2>/dev/null || true
    # Copy VALUE CSV
    cp metrics/"$EXP"/*.csv "$VALIDATION_DIR/" 2>/dev/null || true
    # Copy Legacy plots
    find graph/metrics/"$EXP" -name "*.png" -exec cp {} "$VALIDATION_DIR/" \; 2>/dev/null || true
    # Copy Legacy CSVs
    find metrics/"$EXP" -name "*.csv" -exec cp {} "$VALIDATION_DIR/" \; 2>/dev/null || true
    # Copy PDF report
    cp /scratch/globc/page/idownscale_active/output/"$EXP"/*.pdf "$VALIDATION_DIR/" 2>/dev/null || true
    
    INTEGRITY_LOG="$VALIDATION_DIR/pipeline_integrity.log"
    echo "--- Phase 6: Integrity Check ---" >> "$INTEGRITY_LOG"
    srun $PYTHON bin/utils/check_pipeline_integrity.py --phase 6 --exp "$EXP" | tee -a "$INTEGRITY_LOG" >> "$LOG_DIR/integrity_checks.log" 2>&1
    # Save a copy of the execution log to the validation directory
    cp "$LOG_DIR/execution.log" "$VALIDATION_DIR/run_summary.log"
    
    complete_phase 6
fi

log_progress "=== Workflow Sequence COMPLETE ==="
