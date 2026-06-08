#!/bin/bash
#SBATCH -p grace
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp5_full
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=12:00:00

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUNTIME_ROOT="${IDOWNSCALE_RUNTIME_ROOT:-/scratch/globc/${USER}/idownscale_runtime}"
export IDOWNSCALE_RUNTIME_ROOT="$RUNTIME_ROOT"
export IDOWNSCALE_OUTPUT_DIR="${IDOWNSCALE_OUTPUT_DIR:-${RUNTIME_ROOT}/output}"
export IDOWNSCALE_GRAPHS_DIR="${IDOWNSCALE_GRAPHS_DIR:-${RUNTIME_ROOT}/graphs}"
export IDOWNSCALE_METRICS_DIR="${IDOWNSCALE_METRICS_DIR:-${IDOWNSCALE_OUTPUT_DIR}/metrics}"
export IDOWNSCALE_REGRID_WEIGHTS_DIR="${IDOWNSCALE_REGRID_WEIGHTS_DIR:-${IDOWNSCALE_OUTPUT_DIR}/regrid_weights}"

# ---------------------
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/$EXP/$RUN_ID"
export LOG_DIR
mkdir -p "$LOG_DIR"
# Ensure essential output and data directories exist for first-time runs
mkdir -p "$IDOWNSCALE_OUTPUT_DIR/$EXP/validation" "$IDOWNSCALE_OUTPUT_DIR/$EXP/predictions"
mkdir -p "$IDOWNSCALE_METRICS_DIR/$EXP" "$IDOWNSCALE_GRAPHS_DIR/metrics/$EXP"
mkdir -p "$IDOWNSCALE_OUTPUT_DIR/datasets/dataset_bc"

cd "$REPO_ROOT" || exit 1

module load python/anaconda3.11_arm

CONDA_PREFIX="${IDOWNSCALE_CONDA_PREFIX:-/scratch/globc/${USER}/conda/envs/idownscale_env}"
PYTHON="$CONDA_PREFIX/bin/python"

# Isolate from system Anaconda (PYTHONHOME) and stale ~/.local packages
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Required for xesmf
export ESMFMKFILE="$CONDA_PREFIX/lib/esmf.mk"
export PYTHONUNBUFFERED=1

readarray -t SETTINGS_DATES < <("$PYTHON" - <<'PY'
from iriscc.settings import DATES, DATES_BC_TEST_FUTURE, DATES_BC_TEST_HIST

print(DATES[0].strftime("%Y%m%d"))
print(DATES[-1].strftime("%Y%m%d"))
print(DATES_BC_TEST_HIST[0].strftime("%Y%m%d"))
print(DATES_BC_TEST_HIST[-1].strftime("%Y%m%d"))
print(DATES_BC_TEST_FUTURE[-1].strftime("%Y%m%d"))
PY
)
PHASE1_START_DATE="${PHASE1_START_DATE:-${SETTINGS_DATES[0]}}"
PHASE1_END_DATE="${PHASE1_END_DATE:-${SETTINGS_DATES[1]}}"
START_DATE_INF="${START_DATE_INF:-${SETTINGS_DATES[2]}}"
END_DATE_INF="${END_DATE_INF:-${SETTINGS_DATES[4]}}"
START_DATE_METRICS="${START_DATE_METRICS:-${SETTINGS_DATES[2]}}"
END_DATE_METRICS="${END_DATE_METRICS:-${SETTINGS_DATES[3]}}"
START_DATE_VALUE="${START_DATE_VALUE:-${SETTINGS_DATES[2]}}"
END_DATE_VALUE="${END_DATE_VALUE:-${SETTINGS_DATES[3]}}"

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
    $PYTHON bin/preprocessing/build_dataset.py --exp "$EXP" --start_date "$PHASE1_START_DATE" --end_date "$PHASE1_END_DATE" $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    $PYTHON bin/preprocessing/build_dataset.py --exp "$EXP" --baseline --start_date "$PHASE1_START_DATE" --end_date "$PHASE1_END_DATE" $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
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
    mkdir -p "$IDOWNSCALE_GRAPHS_DIR/metrics/$EXP/$TEST_NAME_$SIMU_TEST"
    srun $PYTHON bin/evaluation/compute_test_metrics_day_fast.py --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" --startdate "$START_DATE_METRICS" --enddate "$END_DATE_METRICS" >> "$LOG_DIR/phase6_metrics.log" 2>&1
    srun $PYTHON bin/evaluation/plot_test_metrics.py --exp "$EXP" --test-name "${TEST_NAME}_${SIMU_TEST}" --scale daily >> "$LOG_DIR/phase6_plots.log" 2>&1
    
    # 6.1 Future Trend Analysis (Qualitative)
    srun $PYTHON bin/evaluation/evaluate_futur_trend.py --exp "$EXP" --ssp "$SSP" --simu "$SIMU" $FORCE_FLAG >> "$LOG_DIR/phase6_trends.log" 2>&1
    
    # 6.2 Historical Validation (Quantitative - VALUE Framework)
    log_progress "--- Phase 6.2: Historical Validation (VALUE) ---"
    srun $PYTHON bin/training/predict_loop.py --startdate "$START_DATE_VALUE" --enddate "$END_DATE_VALUE" --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" $FORCE_FLAG >> "$LOG_DIR/phase6_historical.log" 2>&1
    srun $PYTHON bin/evaluation/compute_value_metrics.py --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" --startdate "$START_DATE_VALUE" --enddate "$END_DATE_VALUE" >> "$LOG_DIR/phase6_value.log" 2>&1
    
    # 6.3 Automated PDF Reporting
    log_progress "--- Phase 6.3: Generating PDF Evaluation Report ---"
    srun $PYTHON bin/evaluation/generate_report.py --exp "$EXP" --test-name "$TEST_NAME" --simu "$SIMU" --ssp "$SSP" >> "$LOG_DIR/phase6_report.log" 2>&1
    
    # 6.4 Consolidate plots in output directory
    log_progress "--- Phase 6.4: Consolidating plots in output directory ---"
    VALIDATION_DIR="$IDOWNSCALE_OUTPUT_DIR/$EXP/validation"
    mkdir -p "$VALIDATION_DIR"
    
    # Copy new SOTA plots
    cp "$IDOWNSCALE_GRAPHS_DIR/metrics/$EXP"/*.png "$VALIDATION_DIR/" 2>/dev/null || true
    # Copy VALUE CSV
    cp "$IDOWNSCALE_METRICS_DIR/$EXP"/*.csv "$VALIDATION_DIR/" 2>/dev/null || true
    # Copy Legacy plots
    find "$IDOWNSCALE_GRAPHS_DIR/metrics/$EXP" -name "*.png" -exec cp {} "$VALIDATION_DIR/" \; 2>/dev/null || true
    # Copy Legacy CSVs
    find "$IDOWNSCALE_METRICS_DIR/$EXP" -name "*.csv" -exec cp {} "$VALIDATION_DIR/" \; 2>/dev/null || true
    # Copy PDF report
    cp "$IDOWNSCALE_OUTPUT_DIR/$EXP"/*.pdf "$VALIDATION_DIR/" 2>/dev/null || true
    
    INTEGRITY_LOG="$VALIDATION_DIR/pipeline_integrity.log"
    echo "--- Phase 6: Integrity Check ---" >> "$INTEGRITY_LOG"
    srun $PYTHON bin/utils/check_pipeline_integrity.py --phase 6 --exp "$EXP" | tee -a "$INTEGRITY_LOG" >> "$LOG_DIR/integrity_checks.log" 2>&1
    # Save a copy of the execution log to the validation directory
    cp "$LOG_DIR/execution.log" "$VALIDATION_DIR/run_summary.log"
    
    complete_phase 6
fi

log_progress "=== Workflow Sequence COMPLETE ==="
