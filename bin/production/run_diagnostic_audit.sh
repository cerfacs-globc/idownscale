#!/bin/bash
# Official Diagnostic Auditor Wrapper
# Usage: ./bin/production/run_diagnostic_audit.sh <p1_path> <p2_path> <idx>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <p1_snapshot_path> <p2_volume_path> <index>"
    exit 1
fi

P1_PATH=$1
P2_PATH=$2
IDX=$3

srun -p grace --gres=gpu:0 --cpus-per-task=16 -t 00:05:00 bash -c "
unset PYTHONHOME
unset PYTHONPATH
module load python/gloenv3.12_arm
export PAGER=cat
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages
export PYTHONNOUSERSITE=1

python3 bin/production/certifier_v86.py \
      --p1_new $P1_PATH \
      --p2_new $P2_PATH \
      --idx $IDX"
