#!/bin/bash
# Official Historical Baseline Census Wrapper
# Purpose: Verify 100% bit-parity of 1980-2014 France Domain snapshots.

srun -p grace --gres=gpu:0 --cpus-per-task=16 -t 00:15:00 bash -c "
unset PYTHONHOME
unset PYTHONPATH
module load python/gloenv3.12_arm
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages
export PYTHONNOUSERSITE=1

python3 bin/production/census_p1.py"
