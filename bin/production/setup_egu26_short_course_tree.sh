#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"

mkdir -p \
  "$ROOT/rawdata/eobs" \
  "$ROOT/rawdata/era5/tas_1d" \
  "$ROOT/rawdata/gcm/CNRM-CM6-1" \
  "$ROOT/rawdata/gcm/CNRM-CM6-1-BC" \
  "$ROOT/scratch/checkpoint_bundles" \
  "$ROOT/idownscale_output/datasets/dataset_exp5_30y" \
  "$ROOT/idownscale_output/metrics/exp5/mean_metrics" \
  "$ROOT/idownscale_output/graph/metrics/exp5" \
  "$ROOT/idownscale_output/prediction" \
  "$ROOT/idownscale_output/runs/exp5"

echo "Created short-course data tree under: $ROOT"
echo
echo "Key directories:"
echo "  $ROOT/rawdata/eobs"
echo "  $ROOT/rawdata/era5/tas_1d"
echo "  $ROOT/rawdata/gcm/CNRM-CM6-1"
echo "  $ROOT/rawdata/gcm/CNRM-CM6-1-BC"
echo "  $ROOT/scratch/checkpoint_bundles"
echo "  $ROOT/idownscale_output/datasets/dataset_exp5_30y"
echo "  $ROOT/idownscale_output/metrics/exp5/mean_metrics"
echo "  $ROOT/idownscale_output/graph/metrics/exp5"
echo "  $ROOT/idownscale_output/prediction"
echo "  $ROOT/idownscale_output/runs/exp5"
