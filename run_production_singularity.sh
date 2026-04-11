#!/bin/bash
#SBATCH -p grace
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=prod_sing_full
#SBATCH --output=prod_sing_full_%j.out
#SBATCH --gres=gpu:1

set -eo pipefail

# --- Singularity Unified Production Protocol ---
# Image: pytorch25.02.sif (Official NVIDIA/Calypso Image)
# Venv: env_idownscale_singularity (Python 3.12, ARM-native)

# 1. Configuration
IMAGE="/softs/local_arm/singularity/images/pytorch25.02.sif"
ENV_PATH="/scratch/globc/page/idownscale_envs/env_idownscale_singularity"
ESMF_PATH="/scratch/globc/page/idownscale_envs/esmf_fixed"

# 2. Environment Variables for Container
export SING_ENV="source $ENV_PATH/bin/activate; \
export PYTHONNOUSERSITE=1; \
unset PYTHONHOME; \
export ESMFMKFILE=\"$ESMF_PATH/lib/esmf_container.mk\"; \
export LD_LIBRARY_PATH=\"$ESMF_PATH/lib:\$LD_LIBRARY_PATH\"; \
export PYTHONPATH=\"$ESMF_PATH/lib/python3.12/site-packages:.:\$PYTHONPATH\""

SINGULARITY_EXEC="singularity run --cleanenv --nv -B /scratch/ $IMAGE bash -c"

echo "[$(date)] --- STARTING FULL SINGULARITY PRODUCTION (PHASES 1-7) ---"

echo "--- PHASE 1: PREPROCESSING (Building Dataset) ---"
$SINGULARITY_EXEC "$SING_ENV; python bin/preprocessing/build_dataset.py --exp exp5"

echo "--- PHASE 2: PREPROCESSING (GCM Mirror) ---"
# Using n_jobs=16 inside container for stability
$SINGULARITY_EXEC "$SING_ENV; python bin/preprocessing/build_dataset_bc.py --simu gcm --var tas --n_jobs 16"

echo "--- PHASE 3: RESTORATION (IBICUS Mirror) ---"
$SINGULARITY_EXEC "$SING_ENV; python bin/preprocessing/bias_correction_ibicus.py --exp exp5 --ssp ssp585 --simu gcm --var tas"

echo "--- PHASE 4: STATISTICS ---"
$SINGULARITY_EXEC "$SING_ENV; python bin/preprocessing/compute_statistics.py --exp exp5"

echo "--- PHASE 5: TRAINING ---"
$SINGULARITY_EXEC "$SING_ENV; python bin/training/train.py"

echo "--- PHASE 6: PREDICTION ---"
$SINGULARITY_EXEC "$SING_ENV; python bin/training/predict_loop.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20000101 --enddate 20141231"
$SINGULARITY_EXEC "$SING_ENV; python bin/training/predict_loop.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20150101 --enddate 21001231"

echo "--- PHASE 7: EVALUATION ---"
$SINGULARITY_EXEC "$SING_ENV; python bin/evaluation/compute_test_metrics_day.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20000101 --enddate 20141231"
$SINGULARITY_EXEC "$SING_ENV; python bin/evaluation/plot_test_metrics.py --exp exp5 --test-name unet --scale daily --simu-test gcm_bc"

echo "[$(date)] --- FULL PRODUCTION COMPLETE ---"
