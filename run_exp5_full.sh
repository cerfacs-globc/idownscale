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

# Permanent environment (ARM native, out-of-repo)
PYTHON="/scratch/globc/page/idownscale_envs/env_idownscale_arm/bin/python"
export PYTHON

# Isolate from system Anaconda (PYTHONHOME) and stale ~/.local packages
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Required for xesmf
# DERIVE path from PYTHON dir
VENV_PATH=$(dirname $(dirname "$PYTHON"))
export ESMFMKFILE="$VENV_PATH/lib/esmf.mk"
export PYTHONUNBUFFERED=1

# --- Modular execution control ---
EXP=${EXP:-exp5}
VAR=${VAR:-tas}
SIMU=${SIMU:-gcm}
SSP=${SSP:-ssp585}
TEST_NAME=${TEST_NAME:-unet_all}
SIMU_TEST=${SIMU_TEST:-gcm}
START_DATE_INF=${START_DATE_INF:-20150101}
END_DATE_INF=${END_DATE_INF:-21001231}

# Use START_PHASE and STOP_PHASE to skip ranges (e.g. START_PHASE=4 ./run_exp5_full.sh)
START_PHASE=${START_PHASE:-1}
STOP_PHASE=${STOP_PHASE:-7}
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
    log_progress "--- Phase 1: Preprocessing START (OPT-4: parallel standard+baseline) ---"
    # OPT-4: Standard and baseline preprocessing are independent — run in parallel.
    WORKERS=${IDOWNSCALE_WORKERS:-1}
    $PYTHON bin/preprocessing/build_dataset.py --exp "$EXP" --workers "$WORKERS" $FORCE_FLAG >> "$LOG_DIR/phase1_std.log" 2>&1 &
    PID_STD=$!
    $PYTHON bin/preprocessing/build_dataset.py --exp "$EXP" --baseline --workers "$WORKERS" $FORCE_FLAG >> "$LOG_DIR/phase1_baseline.log" 2>&1 &
    PID_BASELINE=$!
    wait $PID_STD; RC_STD=$?
    wait $PID_BASELINE; RC_BASELINE=$?
    if [[ $RC_STD -ne 0 || $RC_BASELINE -ne 0 ]]; then
        log_progress "Phase 1 preprocessing FAILED (std=$RC_STD, baseline=$RC_BASELINE)"
        exit 1
    fi
    # Merge phase logs for unified view
    cat "$LOG_DIR/phase1_std.log" "$LOG_DIR/phase1_baseline.log" >> "$LOG_DIR/phase1.log"
    $PYTHON bin/preprocessing/compute_statistics.py --exp "$EXP" $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    $PYTHON bin/preprocessing/compute_statistics_gamma.py --exp "$EXP" $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 1 --exp "$EXP" | tee -a "output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 1
fi

if run_phase 2; then
    log_progress "--- Phase 2: Bias Correction Preprocessing START ---"
    $PYTHON bin/preprocessing/build_dataset_bc.py --simu "$SIMU" --ssp "$SSP" --var "$VAR" $FORCE_FLAG >> "$LOG_DIR/phase2.log" 2>&1
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 2 --exp "$EXP" | tee -a "output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 2
fi

if run_phase 3; then
    log_progress "--- Phase 3: Bias Correction (Ibicus) START ---"
    $PYTHON bin/preprocessing/bias_correction_ibicus.py --exp "$EXP" --ssp "$SSP" --simu "$SIMU" --var "$VAR" $FORCE_FLAG >> "$LOG_DIR/phase3.log" 2>&1
    BC_DATASET_DIR="datasets/dataset_bc/dataset_${EXP}_test_${SIMU}_bc"
    $PYTHON bin/preprocessing/compute_statistics.py --exp "$EXP" --dataset_dir "$BC_DATASET_DIR" $FORCE_FLAG >> "$LOG_DIR/phase3.log" 2>&1
    $PYTHON bin/preprocessing/compute_statistics_gamma.py --exp "$EXP" --dataset_dir "$BC_DATASET_DIR" $FORCE_FLAG >> "$LOG_DIR/phase3.log" 2>&1
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 3 --exp "$EXP" --simu "$SIMU" | tee -a "output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 3
fi

if run_phase 4; then
    log_progress "--- Phase 4: Training START ---"
    $PYTHON bin/training/train.py >> "$LOG_DIR/phase4.log" 2>&1
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 4 --exp "$EXP" | tee -a "output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 4
fi

if run_phase 5; then
    log_progress "--- Phase 5: Inference START ---"
    $PYTHON bin/training/predict_loop.py --startdate "$START_DATE_INF" --enddate "$END_DATE_INF" --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" $FORCE_FLAG >> "$LOG_DIR/phase5.log" 2>&1
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 5 --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" | tee -a "output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 5
fi

if run_phase 6; then
    log_progress "--- Phase 6: Evaluation START ---"
    
    # 6.0 Original Master Evaluation (Legacy - Optimized)
    log_progress "--- Phase 6.0: Original Master Evaluation (Legacy) ---"
    mkdir -p "graph/metrics/$EXP/$TEST_NAME_$SIMU_TEST"
    $PYTHON bin/evaluation/compute_test_metrics_day_fast.py --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" >> "$LOG_DIR/phase6_metrics.log" 2>&1
    $PYTHON bin/evaluation/plot_test_metrics.py --exp "$EXP" --test-name "${TEST_NAME}_${SIMU_TEST}" --scale daily >> "$LOG_DIR/phase6_plots.log" 2>&1
    
    # 6.1 Future Trend Analysis (Qualitative)
    $PYTHON bin/evaluation/evaluate_futur_trend.py --exp "$EXP" --ssp "$SSP" --simu "$SIMU" $FORCE_FLAG >> "$LOG_DIR/phase6_trends.log" 2>&1
    
    # 6.2 Historical Validation (Quantitative - VALUE Framework)
    log_progress "--- Phase 6.2: Historical Validation (VALUE) ---"
    $PYTHON bin/training/predict_loop.py --startdate 20000101 --enddate 20141231 --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" $FORCE_FLAG >> "$LOG_DIR/phase6_historical.log" 2>&1
    $PYTHON bin/evaluation/compute_value_metrics.py --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" >> "$LOG_DIR/phase6_value.log" 2>&1
    
    # 6.3 Automated PDF Reporting
    log_progress "--- Phase 6.3: Generating PDF Evaluation Report ---"
    $PYTHON bin/evaluation/generate_report.py --exp "$EXP" --test-name "$TEST_NAME" --simu "$SIMU" --ssp "$SSP" >> "$LOG_DIR/phase6_report.log" 2>&1
    
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
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 6 --exp "$EXP" | tee -a "$INTEGRITY_LOG" >> "$LOG_DIR/integrity_checks.log" 2>&1
    # Save a copy of the execution log to the validation directory
    cp "$LOG_DIR/execution.log" "$VALIDATION_DIR/run_summary.log"
    
# 7. Comparison Reports
    log_progress "--- Phase 7: Comparison Reports ---"
    REPORT_DIR="output/$EXP/reports"
    mkdir -p "$REPORT_DIR"
    RAW_FILE=$(ls data/GCM/tas_day_*_${SSP}_r1i1p1f2_gcm.nc | head -n 1)
    BC_FILE=$(ls datasets/dataset_bc/tas_day_*_${SSP}_r1i1p1f2_bc.nc | head -n 1)
    AI_FILE=$(ls predictions/exp5/tas_day_*_${SSP}_r1i1p1f2*exp5_unet_all_gcm_bc.nc | head -n 1)
    $PYTHON bin/evaluation/plot_pdf_evolution.py --exp "$EXP" --ssp "$SSP" --raw "$RAW_FILE" --bc "$BC_FILE" --ai "$AI_FILE" >> "$LOG_DIR/phase7.log" 2>&1
    $PYTHON bin/evaluation/generate_comparison_report.py --exp "$EXP" --ssp "$SSP" >> "$LOG_DIR/phase7.log" 2>&1
    mv output/"$EXP"/*.pdf "$REPORT_DIR/" 2>/dev/null || true
    mv output/"$EXP"/validation/*.pdf "$REPORT_DIR/" 2>/dev/null || true
    complete_phase 6
fi

log_progress "=== Workflow Sequence COMPLETE ==="
