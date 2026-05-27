# Environment Setup

This page is the operator-facing environment note for running `idownscale` on
Calypso. It complements `doc/CALYPSO_RUNBOOK.md`, which focuses on the actual
workflow commands.

The short operational summary is:

- GPU is mainly needed for `train` and `predict_loop`
- preprocessing and most evaluation phases can run on CPU
- raw data are discovered from `IDOWNSCALE_RAW_DIR` or `repo/rawdata`
- writable output paths should be overridden if the defaults point to someone
  else's protected project space

## Validated Grace GPU environment

The currently validated Grace GPU path is:

```bash
module load python/gloenv3.12_arm
module load nvidia/cuda/12.4
source /scratch/globc/page/idownscale_envs/production_final_v22_312/bin/activate
unset PYTHONHOME
export PYTHONNOUSERSITE=1
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

By default, the cleaned workflow expects raw data under the repository root:

```bash
/scratch/globc/page/idownscale_rerun/rawdata
```

This comes directly from `iriscc/settings.py`:

- `RAW_DIR = env_path('IDOWNSCALE_RAW_DIR', PROJECT_ROOT / 'rawdata')`

So the discovery rule is simple:

1. use `IDOWNSCALE_RAW_DIR` if set
2. otherwise use `repo/rawdata`

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
cd /scratch/globc/page/idownscale_rerun
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

The generic Grace submitters in `bin/production/` default to that path unless
you override `IDOWNSCALE_VENV_PATH`.

For `globc` CPU fallback, the same workflow commands are valid, but the exact
Python environment may need to be set differently if the Grace venv is not
usable on that partition.

Useful checks:

```bash
bash bin/production/submit_grace_venv_probe.sh
TORCH_VERSION=2.5.1 bash bin/production/submit_grace_torch_version_probe.sh
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
