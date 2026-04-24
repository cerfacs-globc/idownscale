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
#  idownscale_master_v86.sh - End-to-End Production Orchestrator
# ------------------------------------------------------------------------------
set -e

# 1. Environment Synchronization
unset PYTHONHOME
unset PYTHONPATH
module load python/gloenv3.12_arm
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages
export PYTHONNOUSERSITE=1

# 2. Parameters
EXP="exp5"
SSP="ssp585"
SIMU="gcm"

echo "--- STARTING END-TO-END PRODUCTION WORKFLOW (v86.74) ---"

# PHASE 2: Regional Bias Correction (IBICUS Integration)
echo "--- PHASE 2: Synthesizing European 1D Volumes ---"
# This is currently handled by Job 225241. For a fresh run, it would be:
# python3 bin/preprocessing/build_dataset_bc.py --simu $SIMU --exp $EXP --ssp $SSP

# PHASE 3.1: The Discretization Bridge (Parallel regridding to France HR)
echo "--- PHASE 3.1: Discretizing Volumes to Daily Samples ---"
# python3 bin/preprocessing/discretize_bc_parallel.py --exp $EXP --simu ${SIMU}_bc

# PHASE 3.2: High-Resolution Inference (Downscaling)
echo "--- PHASE 3.2: UNet Prediction Loop (Inference) ---"
# HISTORICAL PERIOD
python3 bin/training/predict_loop.py --exp $EXP --test-name unet_all --simu-test ${SIMU}_bc --startdate 19800101 --enddate 20141231
# FUTURE PERIOD
python3 bin/training/predict_loop.py --exp $EXP --test-name unet_all --simu-test ${SIMU}_bc --startdate 20150101 --enddate 21001231

# PHASE 4: Scientific Evaluation (EGU Visualization)
echo "--- PHASE 4: Generating EGU Metrics & Trend Plots ---"
python3 bin/evaluation/evaluate_futur_trend.py --exp $EXP --ssp $SSP --simu $SIMU

echo "--- PRODUCTION WORKFLOW COMPLETE ---"
date
