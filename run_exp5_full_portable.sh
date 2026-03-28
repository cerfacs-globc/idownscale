#!/bin/bash
#SBATCH -p grace
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp5_full
#SBATCH --output=logs/exp5/exp5_full_%j.out
#SBATCH --error=logs/exp5/exp5_full_%j.err
#SBATCH --time=12:00:00

# ------------------------------------------------------------------------------
#  run_exp5_full_portable.sh — Multi-User & Machine Portable Pipeline
#
#  Dynamically discovers the project root and manages code/data separation.
#
#  Usage:
#    sbatch run_exp5_full_portable.sh
#    # Or override Slurm:
#    SBATCH_PARTITION=test sbatch run_exp5_full_portable.sh
# ------------------------------------------------------------------------------

# ── Dynamic Location Discovery ────────────────────────────────────────────────
REPO_ROOT=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
cd "$REPO_ROOT" || exit 1

# ── Environment & Data Roots ──────────────────────────────────────────────────
# Default IDOWNSCALE_DATA_DIR to current directory ($PWD) if not set.
# This allows running from a scratch folder and keeping outputs there.
export IDOWNSCALE_DATA_DIR="${IDOWNSCALE_DATA_DIR:-$PWD}"
# Default IDOWNSCALE_RAW_DIR to the repo's rawdata if not explicitly set.
export IDOWNSCALE_RAW_DIR="${IDOWNSCALE_RAW_DIR:-$REPO_ROOT/rawdata}"
# Default IDOWNSCALE_ENVS to sibling of repo if not set
IDOWNSCALE_ENVS="${IDOWNSCALE_ENVS:-$(dirname "$REPO_ROOT")/idownscale_envs}"

PYTHON="$IDOWNSCALE_ENVS/env_idownscale_arm/bin/python"
export PYTHON
export PYTHONUNBUFFERED=1

# Create essential directories in DATA_DIR
mkdir -p "$IDOWNSCALE_DATA_DIR/logs" "$IDOWNSCALE_DATA_DIR/output"
mkdir -p "$IDOWNSCALE_DATA_DIR/datasets" "$IDOWNSCALE_DATA_DIR/metrics"

# Isolate from system Anaconda/stale ~/.local packages
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# Required for xesmf (derive ESMFMKFILE from VENV path)
VENV_PATH=$(dirname $(dirname "$PYTHON"))
export ESMFMKFILE="$VENV_PATH/lib/esmf.mk"

# --- Modular execution control ---
EXP=${EXP:-exp5}
VAR=${VAR:-tas}
SIMU=${SIMU:-gcm}
SSP=${SSP:-ssp585}
TEST_NAME=${TEST_NAME:-unet_all}
SIMU_TEST=${SIMU_TEST:-gcm}
START_DATE_INF=${START_DATE_INF:-20150101}
END_DATE_INF=${END_DATE_INF:-21001231}

# Use START_PHASE and STOP_PHASE to skip ranges (e.g. START_PHASE=4 ./run_exp5_full_portable.sh)
START_PHASE=${START_PHASE:-1}
STOP_PHASE=${STOP_PHASE:-7}
FORCE=${FORCE:-0}
REGENERATE=${REGENERATE:-0}
MARKER_DIR=".markers"
mkdir -p "$MARKER_DIR"

RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$IDOWNSCALE_DATA_DIR/logs/$EXP/$RUN_ID"
export LOG_DIR
mkdir -p "$LOG_DIR"

FORCE_FLAG=""
if [[ $REGENERATE -eq 1 || $FORCE -eq 1 ]]; then
    FORCE_FLAG="--force"
    [[ $REGENERATE -eq 1 ]] && FORCE=1
fi

log_progress() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/execution.log"
}

run_phase() {
    local phase_num=$1
    local marker="$MARKER_DIR/phase$phase_num.done"
    if [[ $phase_num -lt $START_PHASE || $phase_num -gt $STOP_PHASE ]]; then return 1; fi
    if [[ -f "$marker" && $FORCE -ne 1 ]]; then
        log_progress "Phase $phase_num: SKIPPING (marker found)"
        return 1
    fi
    return 0
}

complete_phase() {
    local phase_num=$1
    touch "$MARKER_DIR/phase$phase_num.done"
    log_progress "Phase $phase_num: COMPLETED"
}

# --- Pipeline Phases ---

if run_phase 1; then
    log_progress "--- Phase 1: Preprocessing START ---"
    $PYTHON bin/preprocessing/build_dataset.py --exp "$EXP" $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    $PYTHON bin/preprocessing/build_dataset.py --exp "$EXP" --baseline $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    $PYTHON bin/preprocessing/compute_statistics.py --exp "$EXP" $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    $PYTHON bin/preprocessing/compute_statistics_gamma.py --exp "$EXP" $FORCE_FLAG >> "$LOG_DIR/phase1.log" 2>&1
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 1 --exp "$EXP" | tee -a "$IDOWNSCALE_DATA_DIR/output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 1
fi

if run_phase 2; then
    log_progress "--- Phase 2: Bias Correction Preprocessing START ---"
    $PYTHON bin/preprocessing/build_dataset_bc.py --simu "$SIMU" --ssp "$SSP" --var "$VAR" $FORCE_FLAG >> "$LOG_DIR/phase2.log" 2>&1
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 2 --exp "$EXP" | tee -a "$IDOWNSCALE_DATA_DIR/output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 2
fi

if run_phase 3; then
    log_progress "--- Phase 3: Bias Correction (Ibicus) START ---"
    $PYTHON bin/preprocessing/bias_correction_ibicus.py --exp "$EXP" --ssp "$SSP" --simu "$SIMU" --var "$VAR" $FORCE_FLAG >> "$LOG_DIR/phase3.log" 2>&1
    BC_DATASET_DIR="$IDOWNSCALE_DATA_DIR/datasets/dataset_bc/dataset_${EXP}_test_${SIMU}_bc"
    $PYTHON bin/preprocessing/compute_statistics.py --exp "$EXP" --dataset_dir "$BC_DATASET_DIR" $FORCE_FLAG >> "$LOG_DIR/phase3.log" 2>&1
    $PYTHON bin/preprocessing/compute_statistics_gamma.py --exp "$EXP" --dataset_dir "$BC_DATASET_DIR" $FORCE_FLAG >> "$LOG_DIR/phase3.log" 2>&1
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 3 --exp "$EXP" --simu "$SIMU" | tee -a "$IDOWNSCALE_DATA_DIR/output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 3
fi

if run_phase 4; then
    log_progress "--- Phase 4: Training START ---"
    $PYTHON bin/training/train.py >> "$LOG_DIR/phase4.log" 2>&1
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 4 --exp "$EXP" | tee -a "$IDOWNSCALE_DATA_DIR/output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 4
fi

if run_phase 5; then
    log_progress "--- Phase 5: Inference START ---"
    $PYTHON bin/training/predict_loop.py --startdate "$START_DATE_INF" --enddate "$END_DATE_INF" --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" $FORCE_FLAG >> "$LOG_DIR/phase5.log" 2>&1
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 5 --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" | tee -a "$IDOWNSCALE_DATA_DIR/output/$EXP/validation/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    complete_phase 5
fi

if run_phase 6; then
    log_progress "--- Phase 6: Evaluation START ---"
    VALIDATION_DIR="$IDOWNSCALE_DATA_DIR/output/$EXP/validation"
    mkdir -p "$VALIDATION_DIR"
    
    $PYTHON bin/evaluation/compute_test_metrics_day_fast.py --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" >> "$LOG_DIR/phase6_metrics.log" 2>&1
    $PYTHON bin/evaluation/plot_test_metrics.py --exp "$EXP" --test-name "${TEST_NAME}_${SIMU_TEST}" --scale daily >> "$LOG_DIR/phase6_plots.log" 2>&1
    $PYTHON bin/evaluation/evaluate_futur_trend.py --exp "$EXP" --ssp "$SSP" --simu "$SIMU" $FORCE_FLAG >> "$LOG_DIR/phase6_trends.log" 2>&1
    
    $PYTHON bin/training/predict_loop.py --startdate 20000101 --enddate 20141231 --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" $FORCE_FLAG >> "$LOG_DIR/phase6_historical.log" 2>&1
    $PYTHON bin/evaluation/compute_value_metrics.py --exp "$EXP" --test-name "$TEST_NAME" --simu-test "$SIMU_TEST" >> "$LOG_DIR/phase6_value.log" 2>&1
    $PYTHON bin/evaluation/generate_report.py --exp "$EXP" --test-name "$TEST_NAME" --simu "$SIMU" --ssp "$SSP" >> "$LOG_DIR/phase6_report.log" 2>&1
    
    # Consolidate in DATA_DIR
    find graph/metrics/"$EXP" -name "*.png" -exec cp {} "$VALIDATION_DIR/" \; 2>/dev/null || true
    find metrics/"$EXP" -name "*.csv" -exec cp {} "$VALIDATION_DIR/" \; 2>/dev/null || true
    cp "$IDOWNSCALE_DATA_DIR/output/$EXP"/*.pdf "$VALIDATION_DIR/" 2>/dev/null || true
    
    $PYTHON bin/utils/check_pipeline_integrity.py --phase 6 --exp "$EXP" | tee -a "$VALIDATION_DIR/pipeline_integrity.log" >> "$LOG_DIR/integrity_checks.log" 2>&1
    cp "$LOG_DIR/execution.log" "$VALIDATION_DIR/run_summary.log"
    complete_phase 6
fi

if run_phase 7; then
    log_progress "--- Phase 7: Comparison Reports ---"
    REPORT_DIR="$IDOWNSCALE_DATA_DIR/output/$EXP/reports"
    mkdir -p "$REPORT_DIR"
    RAW_FILE=$(ls "$IDOWNSCALE_DATA_DIR/rawdata/gcm/CNRM-CM6-1/historical/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc" | head -n 1) # Fallback to repo if raw not moved
    BC_FILE=$(ls "$IDOWNSCALE_DATA_DIR/datasets/dataset_bc/tas_day_*_${SSP}_r1i1p1f2_bc.nc" | head -n 1)
    AI_FILE=$(ls "$IDOWNSCALE_DATA_DIR/output/$EXP/predictions/tas_day_*_${SSP}_r1i1p1f2*exp5_unet_all_gcm_bc.nc" | head -n 1)
    $PYTHON bin/evaluation/plot_pdf_evolution.py --exp "$EXP" --ssp "$SSP" --raw "$RAW_FILE" --bc "$BC_FILE" --ai "$AI_FILE" >> "$LOG_DIR/phase7.log" 2>&1
    $PYTHON bin/evaluation/generate_comparison_report.py --exp "$EXP" --ssp "$SSP" >> "$LOG_DIR/phase7.log" 2>&1
    mv "$IDOWNSCALE_DATA_DIR/output/$EXP"/*.pdf "$REPORT_DIR/" 2>/dev/null || true
    complete_phase 7
fi

log_progress "=== Workflow Sequence COMPLETE ==="
