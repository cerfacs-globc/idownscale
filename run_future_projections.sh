#!/bin/bash
#SBATCH -J production_future
#SBATCH -p grace
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 11:59:00
#SBATCH -o production_future_%j.out

set -e  # Fail on any error
set -u  # Fail on unset variables

# Environment configuration
ENV_NAME="env_idownscale_singularity"
ENVS_ROOT="/scratch/globc/page/idownscale_envs"
ENV_PATH="${ENVS_ROOT}/${ENV_NAME}"
ESMF_PATH="${ENVS_ROOT}/esmf_fixed"
IMAGE="/softs/local_arm/singularity/images/pytorch25.02.sif"

# Standardized env-setup string
SING_ENV="source ${ENV_PATH}/bin/activate; export PYTHONNOUSERSITE=1; unset PYTHONHOME; export ESMFMKFILE=\"${ESMF_PATH}/lib/esmf_container.mk\"; export LD_LIBRARY_PATH=\"${ESMF_PATH}/lib:\$LD_LIBRARY_PATH\"; export PYTHONPATH=\"${ESMF_PATH}/lib/python3.12/site-packages:.:\$PYTHONPATH\""
SINGULARITY="singularity run --cleanenv --nv -B /scratch/ ${IMAGE} bash -c"

echo "--- STARTING PRODUCTION TRAINING & FUTURE PROJECTIONS ---"

echo "--- PHASE 5: TRAINING (60 EPOCHS) ---"
# Training automatically uses DATES_EXP5_TRAIN (1989-2003) as defined in iriscc/settings.py
${SINGULARITY} "${SING_ENV}; python -u bin/training/train.py"

echo "--- PHASE 6: INFERENCE (2000-2014 & Future) ---"
# Inference window for verification (2000-2014)
${SINGULARITY} "${SING_ENV}; python -u bin/training/predict_loop.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20000101 --enddate 20141231"

echo "--- PHASE 7: EVALUATION (2000-2014) ---"
# Evaluation window consistent with inference for validation
${SINGULARITY} "${SING_ENV}; python -u bin/evaluation/compute_test_metrics_day.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20000101 --enddate 20141231"
${SINGULARITY} "${SING_ENV}; python -u bin/evaluation/plot_test_metrics.py --exp exp5 --test-name unet_gcm_bc --scale daily --startdate 20000101 --enddate 20141231"

echo "--- PRODUCTION WORKFLOW COMPLETE ---"
