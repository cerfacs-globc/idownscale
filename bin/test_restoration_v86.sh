#!/bin/bash
#SBATCH --job-name=v86_restoration_test
#SBATCH --partition=grace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out

set -e
module purge
module load python/gloenv3.12_arm

# Environment Sync
unset PYTHONHOME
export PYTHONPATH=/scratch/globc/page/lib_idownscale_phase2:/scratch/globc/page/lib_idownscale_phase2/lib/python3.12/site-packages:/scratch/globc/page/lib_idownscale_phase2/lib64/python3.12/site-packages:$PYTHONPATH
export PYTHONNOUSERSITE=1

echo "--- RESTORATION STRESS TEST: 120-YEAR PIPELINE (v86.74) ---"
# DIAGNOSTIC: Check temporal properties of the future file
python3 -c "import xarray as xr; ds=xr.open_dataset('rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20150101-21001231.nc'); print('--- FUTURE TIME SAMPLE ---'); print(ds.time.values[:3]); print(ds.time.dt.year.values[:3]); print(ds.time.dt.month.values[:3]); print(ds.time.dt.day.values[:3])"

python3 bin/preprocessing/build_dataset_bc.py \
    --simu gcm \
    --exp exp5 \
    --var tas \
    --test \
    --output_dir /scratch/globc/page/idownscale_output/full_pipeline_test/

echo "--- TEST COMPLETE ---"
ls -lh /scratch/globc/page/idownscale_output/full_pipeline_test/
