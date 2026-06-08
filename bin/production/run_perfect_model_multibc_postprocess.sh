#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_ROOT_DEFAULT="/scratch/globc/${USER}/idownscale_runtime"
export IDOWNSCALE_RUNTIME_ROOT="${IDOWNSCALE_RUNTIME_ROOT:-${RUNTIME_ROOT_DEFAULT}}"
export IDOWNSCALE_OUTPUT_DIR="${IDOWNSCALE_OUTPUT_DIR:-${IDOWNSCALE_RUNTIME_ROOT}/output}"

EXP="${EXP:-perfect_model_rcm}"
SIMU_TEST="${SIMU_TEST:-rcm}"
VAR="${VAR:-tas}"
RAW_SAMPLE_DIR="${RAW_SAMPLE_DIR:-${IDOWNSCALE_OUTPUT_DIR}/datasets/dataset_bc/dataset_perfect_model_rcm_all_windows_rcm_raw}"
COMPARISON_DIR="${COMPARISON_DIR:-${IDOWNSCALE_OUTPUT_DIR}/metrics/${EXP}/comparison_tables}"
CHUNKS_DIR="${CHUNKS_DIR:-${COMPARISON_DIR}/chunks}"

mkdir -p "${COMPARISON_DIR}" "${CHUNKS_DIR}"

MODELS=(
  "unet_outputnorm_perfect_model_rcm"
  "unet_perfect_model_rcm"
  "unet_rep3_perfect_model_rcm"
  "miniunet_perfect_model_rcm"
  "unet_seed2_perfect_model_rcm"
)

EXACT_WINDOWS=(
  "20000101_20141231"
  "20900101_21001231"
)

for model in "${MODELS[@]}"; do
  for window in "${EXACT_WINDOWS[@]}"; do
    start="${window%_*}"
    end="${window#*_}"

    "${PYTHON_BIN}" bin/evaluation/compare_perfect_model_predictions_vs_truth.py \
      --exp "${EXP}" \
      --test-name "${model}" \
      --simu-test "${SIMU_TEST}" \
      --var "${VAR}" \
      --startdate "${start}" \
      --enddate "${end}" \
      --sample-dir "${RAW_SAMPLE_DIR}" \
      --raw-sample-dir "${RAW_SAMPLE_DIR}" \
      --output-dir "${CHUNKS_DIR}" \
      --stem-suffix "_${window}" \
      --include-bc-baseline

    "${PYTHON_BIN}" bin/evaluation/compare_perfect_model_predictions_vs_truth.py \
      --exp "${EXP}" \
      --test-name "${model}" \
      --simu-test "${SIMU_TEST}" \
      --var "${VAR}" \
      --startdate "${start}" \
      --enddate "${end}" \
      --sample-dir "${RAW_SAMPLE_DIR}" \
      --raw-sample-dir "${RAW_SAMPLE_DIR}" \
      --output-dir "${CHUNKS_DIR}" \
      --stem-suffix "_${window}" \
      --include-bc-baseline \
      --bc-tag sbck_cdft \
      --bc-model-label bc_baseline_sbck_cdft
  done
done

AGG_CMD=(
  "${PYTHON_BIN}" bin/evaluation/aggregate_perfect_model_comparisons.py
  --exp "${EXP}"
  --simu-test "${SIMU_TEST}"
  --var "${VAR}"
  --chunks-dir "${CHUNKS_DIR}"
  --output-dir "${COMPARISON_DIR}"
)
for model in "${MODELS[@]}"; do
  AGG_CMD+=(--test-name "${model}")
done
"${AGG_CMD[@]}"

"${PYTHON_BIN}" bin/evaluation/plot_perfect_model_comparison.py \
  --exp "${EXP}" \
  --simu-test "${SIMU_TEST}" \
  --var "${VAR}" \
  --input-csv "${COMPARISON_DIR}/perfect_model_predictions_vs_truth_${EXP}_combined_${SIMU_TEST}.csv"

"${PYTHON_BIN}" bin/evaluation/plot_perfect_model_distribution_pdf.py \
  --exp "${EXP}" \
  --simu-test "${SIMU_TEST}" \
  --var "${VAR}" \
  --sample-dir "${RAW_SAMPLE_DIR}" \
  --raw-sample-dir "${RAW_SAMPLE_DIR}" \
  --input-csv "${COMPARISON_DIR}/perfect_model_predictions_vs_truth_${EXP}_combined_${SIMU_TEST}.csv" \
  --window 20000101_20141231 \
  --window 20900101_21001231

"${PYTHON_BIN}" bin/evaluation/compare_perfect_model_climate_signal.py \
  --exp "${EXP}" \
  --simu-test "${SIMU_TEST}" \
  --var "${VAR}" \
  --sample-dir "${RAW_SAMPLE_DIR}" \
  --raw-sample-dir "${RAW_SAMPLE_DIR}" \
  --input-csv "${COMPARISON_DIR}/perfect_model_predictions_vs_truth_${EXP}_combined_${SIMU_TEST}.csv" \
  --reference-window 19810101_20101231 \
  --future-window 20800101_21001231

"${PYTHON_BIN}" bin/evaluation/compare_perfect_model_window_statistics.py \
  --exp "${EXP}" \
  --simu-test "${SIMU_TEST}" \
  --var "${VAR}" \
  --sample-dir "${RAW_SAMPLE_DIR}" \
  --raw-sample-dir "${RAW_SAMPLE_DIR}" \
  --input-csv "${COMPARISON_DIR}/perfect_model_predictions_vs_truth_${EXP}_combined_${SIMU_TEST}.csv"
