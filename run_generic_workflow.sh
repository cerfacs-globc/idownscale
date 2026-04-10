#!/bin/bash
# run_generic_workflow.sh: Sequential pipeline execution without SLURM/Singularity

set -eo pipefail

EXP="exp5"
START_HIST="19890101"
END_HIST="20031231"
START_TEST="20100101"
END_TEST="20141231"

echo "=== PHASE 1: PREPROCESSING (Building Dataset) ==="
python bin/preprocessing/build_dataset.py --exp $EXP

echo "=== PHASE 4: STATISTICS (Computing Norm Factors) ==="
python bin/preprocessing/compute_statistics.py --exp $EXP

echo "=== PHASE 5: TRAINING (U-Net Optimization) ==="
python bin/training/train.py

echo "=== PHASE 6: PREDICTION (Historical Holdout) ==="
python bin/training/predict_loop.py --exp $EXP --test-name unet --simu-test gcm_bc --startdate $START_TEST --enddate $END_TEST

echo "=== PHASE 7: EVALUATION (Metrics Analysis) ==="
python bin/evaluation/compute_test_metrics_day.py --exp $EXP --test-name unet --simu-test gcm_bc --startdate $START_TEST --enddate $END_TEST

echo "=== Workflow Complete ==="
