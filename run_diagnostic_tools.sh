#!/bin/bash
#SBATCH -J production_diag
#SBATCH -p grace
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 01:59:00
#SBATCH -o production_diag_%j.out

set -e
set -u

# Environment configuration
ENV_NAME="env_idownscale_singularity"
ENVS_ROOT="/scratch/globc/page/idownscale_envs"
ENV_PATH="${ENVS_ROOT}/${ENV_NAME}"
ESMF_PATH="${ENVS_ROOT}/esmf_fixed"
IMAGE="/softs/local_arm/singularity/images/pytorch25.02.sif"

# Standardized env-setup string
SING_ENV="source ${ENV_PATH}/bin/activate; export PYTHONNOUSERSITE=1; unset PYTHONHOME; export ESMFMKFILE=\"${ESMF_PATH}/lib/esmf_container.mk\"; export LD_LIBRARY_PATH=\"${ESMF_PATH}/lib:\$LD_LIBRARY_PATH\"; export PYTHONPATH=\"${ESMF_PATH}/lib/python3.12/site-packages:.:\$PYTHONPATH\""
SINGULARITY="singularity run --cleanenv --nv -B /scratch/ ${IMAGE} bash -c"

echo "--- STARTING PRODUCTION DIAGNOSTIC ---"

# Quick diagnostic check
echo "--- PHASE 5: TRAINING (SHORT DIAGNOSTIC RUN) ---"
${SINGULARITY} "${SING_ENV}; python -u bin/training/train.py"

echo "--- PHASE 7: EVALUATION (DIAGNOSTIC) ---"
# Check metrics on the last checkpoint
${SINGULARITY} "${SING_ENV}; python -u bin/evaluation/compute_test_metrics_day.py --exp exp5 --test-name unet --simu-test gcm_bc --startdate 20000101 --enddate 20141231"

echo "--- DIAGNOSTIC COMPLETE ---"
