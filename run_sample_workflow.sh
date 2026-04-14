#!/bin/bash
#SBATCH -p grace
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --time=12:00:00
#SBATCH --job-name=sample_workflow
#SBATCH --output=sample_workflow_%j.out
#SBATCH --gres=gpu:1

set -eo pipefail

# --- Configuration ---
ENV_NAME="env_idownscale_singularity"
ENVS_ROOT="/scratch/globc/page/idownscale_envs"
ENV_PATH="${ENVS_ROOT}/${ENV_NAME}"
ESMF_PATH="${ENVS_ROOT}/esmf_fixed"
IMAGE="/softs/local_arm/singularity/images/pytorch25.02.sif"

# Shared env-setup string for Singularity (No leading/trailing newlines to avoid syntax errors)
SING_ENV="source ${ENV_PATH}/bin/activate; export PYTHONNOUSERSITE=1; unset PYTHONHOME; export ESMFMKFILE=\"${ESMF_PATH}/lib/esmf_container.mk\"; export LD_LIBRARY_PATH=\"${ESMF_PATH}/lib:\$LD_LIBRARY_PATH\"; export PYTHONPATH=\"${ESMF_PATH}/lib/python3.12/site-packages:.:\$PYTHONPATH\""

SINGULARITY="singularity run --cleanenv --nv -B /scratch/ ${IMAGE} bash -c"

echo "--- STARTING 1-YEAR SAMPLE WORKFLOW ---"

echo "--- PHASE 1: ERA5 GENERATION ---"
${SINGULARITY} "${SING_ENV}; python -u bin/preprocessing/build_dataset.py --exp exp5 --n_jobs 32"

echo "--- PHASE 2: GCM GENERATION ---"
${SINGULARITY} "${SING_ENV}; python -u bin/preprocessing/build_dataset_bc.py --simu gcm --var tas --n_jobs 32"

echo "--- PHASE 3: BIAS CORRECTION ---"
${SINGULARITY} "${SING_ENV}; python -u bin/preprocessing/bias_correction_ibicus.py --exp exp5 --ssp ssp585 --simu gcm --var tas"

echo "--- PHASE 4: STATISTICS ---"
${SINGULARITY} "${SING_ENV}; python -u bin/preprocessing/compute_statistics.py --exp exp5"

echo "--- PHASE 5: TRAINING ---"
${SINGULARITY} "${SING_ENV}; python -u bin/training/train.py --epochs 10"

echo "--- PHASE 6: PREDICTION ---"
${SINGULARITY} "${SING_ENV}; python -u bin/training/predict_loop.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20000101 --enddate 20001231"

echo "--- PHASE 7: EVALUATION ---"
${SINGULARITY} "${SING_ENV}; python -u bin/evaluation/compute_test_metrics_day.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20000101 --enddate 20001231"
${SINGULARITY} "${SING_ENV}; python -u bin/evaluation/plot_test_metrics.py --exp exp5 --test-name unet --scale daily --startdate 20000101 --enddate 20001231"

echo "--- 1-YEAR SAMPLE WORKFLOW COMPLETE ---"
