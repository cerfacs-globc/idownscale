#!/bin/bash
#SBATCH -p grace
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --time=12:00:00
#SBATCH --job-name=garcia_clean
#SBATCH --output=garcia_clean_%j.out
#SBATCH --gres=gpu:1

# --- Clean Room Execution Protocol ---
# Total isolation from active/opt branches. 
# Replicates Zoe Garcia 0.1.0 release results.

export PYTHONNOUSERSITE=1
export PYTHONPATH=.
GARCIA_PYTHON="/scratch/globc/page/idownscale_envs/env_idownscale_arm/bin/python3.11 -I -u"

echo "--- PHASE 1: PREPROCESSING (Building Dataset) ---"
$GARCIA_PYTHON bin/preprocessing/build_dataset.py --exp exp5

echo "--- PHASE 2: PREPROCESSING (GCM Mirror) ---"
$GARCIA_PYTHON bin/preprocessing/build_dataset_bc.py --simu gcm --var tas --n_jobs 16

echo "--- PHASE 3: RESTORATION (IBICUS Mirror) ---"
$GARCIA_PYTHON bin/preprocessing/bias_correction_ibicus.py --exp exp5 --ssp ssp585 --simu gcm --var tas

echo "--- PHASE 4: STATISTICS ---"
$GARCIA_PYTHON bin/preprocessing/compute_statistics.py --exp exp5

echo "--- PHASE 5: TRAINING ---"
$GARCIA_PYTHON bin/training/train.py

echo "--- PHASE 6: PREDICTION ---"
$GARCIA_PYTHON bin/training/predict_loop.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20000101 --enddate 20141231
$GARCIA_PYTHON bin/training/predict_loop.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20150101 --enddate 21001231

echo "--- PHASE 7: EVALUATION ---"
$GARCIA_PYTHON bin/evaluation/compute_test_metrics_day.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20000101 --enddate 20141231
$GARCIA_PYTHON bin/evaluation/plot_test_metrics.py --exp exp5 --test-name unet --scale daily

echo "--- CLEAN ROOM PRODUCTION COMPLETE ---"
