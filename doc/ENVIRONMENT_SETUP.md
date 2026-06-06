# Environment Setup

This page is the operator-facing environment note for running `idownscale` on
Calypso. It complements `doc/CALYPSO_RUNBOOK.md`, which focuses on the actual
workflow commands.

The short operational summary is:

- GPU is mainly needed for `train` and `predict_loop`
- preprocessing and most evaluation phases can run on CPU
- raw data are discovered from `IDOWNSCALE_RAW_DIR`, otherwise from
  `repo/rawdata` if that directory exists, otherwise from
  `IDOWNSCALE_RUNTIME_ROOT/rawdata`
- writable output paths should be overridden if the defaults point to someone
  else's protected project space
- the safest layout is `code in $HOME`, `runtime/data/output on scratch`

## Validated Grace GPU environment

The currently validated Grace GPU path is:

```bash
module load python/gloenv3.12_arm
module load nvidia/cuda/12.4
source /scratch/globc/page/idownscale_envs/production_final_v22_312/bin/activate
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export ESMFMKFILE=/softs/local_arm/Anaconda/2024.02-1/envs/gloenv_py3.11_arm/lib/esmf.mk
```

This is the environment that was proven to work for the Calypso Grace parity
recovery and for GPU training on the `grace` partition.

## Runtime paths

The workflow reads its main paths from environment variables. If you do not set
them, the defaults come from `iriscc/settings.py`.

The most important ones are:

```bash
export IDOWNSCALE_RAW_DIR=/path/to/rawdata
export IDOWNSCALE_OUTPUT_DIR=/path/to/output
export IDOWNSCALE_REGRID_WEIGHTS_DIR=/path/to/output/weights
export IDOWNSCALE_RUNS_DIR=/path/to/output/runs
export IDOWNSCALE_PREDICTION_DIR=/path/to/output/prediction
export IDOWNSCALE_METRICS_DIR=/path/to/output/metrics
export IDOWNSCALE_EXP5_ARCHIVE_DATASET_DIR=/path/to/archive/dataset_exp5_30y
export IDOWNSCALE_GCM_BC_DIR=/path/to/writable/gcm_bc
export IDOWNSCALE_RCM_BC_DIR=/path/to/writable/rcm_bc
```

Recommended split layout when the repository lives in backed-up `home`:

```bash
export IDOWNSCALE_RUNTIME_ROOT=/scratch/globc/$USER/idownscale_runtime
export IDOWNSCALE_RAW_DIR=$IDOWNSCALE_RUNTIME_ROOT/rawdata
export IDOWNSCALE_OUTPUT_DIR=$IDOWNSCALE_RUNTIME_ROOT/idownscale_output
export IDOWNSCALE_REGRID_WEIGHTS_DIR=$IDOWNSCALE_OUTPUT_DIR/weights
export IDOWNSCALE_DATASET_DIR=$IDOWNSCALE_OUTPUT_DIR/datasets
export IDOWNSCALE_RUNS_DIR=$IDOWNSCALE_OUTPUT_DIR/runs
export IDOWNSCALE_PREDICTION_DIR=$IDOWNSCALE_OUTPUT_DIR/prediction
export IDOWNSCALE_METRICS_DIR=$IDOWNSCALE_OUTPUT_DIR/metrics
export IDOWNSCALE_GRAPHS_DIR=$IDOWNSCALE_OUTPUT_DIR/graph
```

On Calypso, the intended defaults are usually:

```bash
export IDOWNSCALE_RAW_DIR=/scratch/globc/page/idownscale_rerun/rawdata
export IDOWNSCALE_OUTPUT_DIR=/gpfs-calypso/scratch/globc/page/idownscale_output
export IDOWNSCALE_REGRID_WEIGHTS_DIR=/gpfs-calypso/scratch/globc/page/idownscale_output/weights
```

If writes must not go into your project-owned output tree, override at least:

```bash
export IDOWNSCALE_OUTPUT_DIR=/gpfs-calypso/scratch/<their-user>/idownscale_output
export IDOWNSCALE_REGRID_WEIGHTS_DIR=$IDOWNSCALE_OUTPUT_DIR/weights
export IDOWNSCALE_DATASET_DIR=$IDOWNSCALE_OUTPUT_DIR/datasets
export IDOWNSCALE_DATASET_BC_DIR=$IDOWNSCALE_OUTPUT_DIR/datasets/dataset_bc
export IDOWNSCALE_RUNS_DIR=$IDOWNSCALE_OUTPUT_DIR/runs
export IDOWNSCALE_PREDICTION_DIR=$IDOWNSCALE_OUTPUT_DIR/prediction
export IDOWNSCALE_METRICS_DIR=$IDOWNSCALE_OUTPUT_DIR/metrics
export IDOWNSCALE_GRAPHS_DIR=$IDOWNSCALE_OUTPUT_DIR/graph
export IDOWNSCALE_GCM_BC_DIR=$IDOWNSCALE_OUTPUT_DIR/gcm_bc
export IDOWNSCALE_RCM_BC_DIR=$IDOWNSCALE_OUTPUT_DIR/rcm_bc
```

## Raw-data location

The most common source of confusion is `rawdata/`.

If `repo/rawdata` exists, the cleaned workflow can still read raw data there.
If it does not exist, the default fallback is:

```bash
$IDOWNSCALE_RUNTIME_ROOT/rawdata
```

This comes directly from `iriscc/settings.py`:

- `RAW_DIR = IDOWNSCALE_RAW_DIR if set`
- otherwise `repo/rawdata` if that directory exists
- otherwise `IDOWNSCALE_RUNTIME_ROOT/rawdata`
- `OUTPUT_DIR = IDOWNSCALE_OUTPUT_DIR if set`
- otherwise `IDOWNSCALE_RUNTIME_ROOT/idownscale_output`

So the discovery rule is simple:

1. use `IDOWNSCALE_RAW_DIR` if set
2. otherwise use `repo/rawdata` if it exists
3. otherwise use `IDOWNSCALE_RUNTIME_ROOT/rawdata`

That can be:

1. a real directory populated inside the repo tree
2. a symlink named `rawdata` pointing to shared storage
3. an external directory provided through `IDOWNSCALE_RAW_DIR`

If the raw data are stored elsewhere, either:

```bash
export IDOWNSCALE_RAW_DIR=/shared/location/rawdata
```

or:

```bash
cd /home/globc/$USER/src/idownscale
ln -s /shared/location/rawdata rawdata
```

Important write-path nuance:

- `prep_phase1` writes prepared France target files under `rawdata/eobs`
- this means `rawdata/eobs` must be writable for that phase
- if the engineer cannot write there, they should either work with a writable
  rawdata copy/mirror or skip `prep_phase1` and reuse the prepared France files

## Minimum raw-data tree for `exp5`

At minimum, the workflow expects these top-level subdirectories:

```text
rawdata/
├── eobs/
├── era5/
├── gcm/
└── topography/
```

Commonly used `exp5` files include:

```text
rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc
rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc
rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc
rawdata/era5/orography_ERA5.nc
rawdata/gcm/CNRM-CM6-1/tas_day_CNRM-CM6-1_historical_r1i1p1f2_gr_18500101-20141231.nc
```

If those files already exist and are readable, no code modification is needed.
The engineer only needs the correct path wiring.

The full upstream-data story is documented in:

- `docs/egu26_short_course/HOW_TO_FETCH_UPSTREAM_DATA.md`
- `docs/egu26_short_course/DATA_SETUP_QUICKSTART.md`

## Reusing the known-good venv

For Grace GPU runs, reuse:

```bash
/scratch/globc/page/idownscale_envs/production_final_v22_312
```

Repair note from 2026-06-01:

- this env had drifted into an inconsistent mixed stack
- the repaired working core is now:
  - `numpy 1.26.4`
  - `scipy 1.12.0`
  - `xarray 2024.7.0`
  - `dask 2024.5.2`
  - `xesmf 0.8.5`
  - `torch 2.5.1`
  - `pytorch_lightning 2.4.0`
  - `ibicus 1.1.1`
- on Grace, `xesmf` still requires:
  - `ESMFMKFILE=/softs/local_arm/Anaconda/2024.02-1/envs/gloenv_py3.11_arm/lib/esmf.mk`
- default wrapper validation now passes through:
  - `bin/production/run_in_grace_env.sh`
  - `bin/production/run_exp5_workflow_grace.sh`
  - `bin/production/submit_exp5_workflow_grace.sh`
- verified on-arm via default wrapper probe job `272675`

The generic Grace submitters in `bin/production/` default to that path unless
you override `IDOWNSCALE_VENV_PATH`.

If that venv is inconsistent on a given node, use the mixed Calypso fallback
that keeps the Grace module stack and switches to the documented site-packages
overlay:

```bash
bash bin/production/run_in_calypso_mixed_env.sh python -c "import xarray, xesmf, torch, ibicus; print('ok')"
```

For `globc` CPU fallback, the same workflow commands are valid, but the exact
Python environment may need to be set differently if the Grace venv is not
usable on that partition.

Validated `globc` CPU fallback:

```bash
/scratch/globc/page/idownscale_envs/globc_cpu_py312_v2
```

Important shell hygiene on Calypso login nodes:

- inherited `PYTHONHOME=/softs/Anaconda/...` can break alternate envs with
  `ModuleNotFoundError: encodings`
- use the wrapper below, or manually `unset PYTHONHOME` and `unset PYTHONPATH`
  before invoking that CPU env

Recommended wrapper:

```bash
bash bin/production/run_in_globc_cpu_env.sh -c "import xarray, xesmf, ibicus; print('ok')"
```

Useful checks:

```bash
bash bin/production/submit_grace_venv_probe.sh
TORCH_VERSION=2.5.1 bash bin/production/submit_grace_torch_version_probe.sh
bash bin/production/run_in_grace_env.sh python -c "import numpy, scipy, xarray, dask, xesmf, torch, ibicus; print('ok')"
```

## Rebuilding the environment

If the validated venv disappears or must be rebuilt, the recovery note is:

- `doc/GRACE_TRAINING_ENGINEER_NOTE.md`

The short version is:

1. start from `python/gloenv3.12_arm`
2. load `nvidia/cuda/12.4`
3. create a clean Python 3.12 environment
4. install PyTorch from the `cu124` index
5. then install the rest of the stack

For Grace GPU, this route is validated. For `globc` CPU fallback, the workflow
commands remain the same, but the exact Python environment should be treated as
site-specific unless separately validated on that partition.

## Settings to inspect

The main settings live in:

```bash
iriscc/settings.py
```

Inspect:

- path discovery and defaults near the top of the file
- `CONFIG['exp5']` for the experiment geometry and target files
- `bin/production/run_exp5_workflow.py` for the phase orchestration and required
  arguments per phase
