# Calypso Runbook

This is the operator runbook for rerunning `idownscale` on Calypso.

It answers the practical questions an engineer should not have to guess:

- where raw data must be visible
- what the master workflow script is
- how to run on Grace GPU
- how to run on `globc` CPU if GPU is unavailable
- when GPU is actually needed
- how to run specific phases
- how to reuse the validated Python environment

## 1. Canonical repo and output locations

Current Calypso repo root:

```bash
/scratch/globc/page/idownscale_rerun
```

Current output root:

```bash
/gpfs-calypso/scratch/globc/page/output
```

Work from the repo root:

```bash
cd /scratch/globc/page/idownscale_rerun
```

## 2. The master workflow entrypoint

The master script is:

```bash
bin/production/run_obs_workflow.py
```

That script orchestrates the cleaned observation-target workflow phases.

Supported phases:

- `prep_phase1`
- `phase1`
- `stats`
- `bc_dataset`
- `bc_apply`
- `train`
- `raw_dataset`
- `pp_dataset`
- `predict_loop`
- `metrics_day`
- `metrics_month`
- `value_metrics`
- `plot_metrics_day`
- `plot_metrics_month`

Important phase constraints from the workflow code:

- `train` requires `--test-name`
- `predict_loop` requires `--test-name`
- `metrics_day` requires `--test-name`
- `metrics_month` requires `--test-name`
- `value_metrics` requires `--test-name`
- `plot_metrics_day` requires `--test-name`
- `plot_metrics_month` requires `--test-name`
- `raw_dataset` currently supports only `--simu gcm`
- `pp_dataset` currently supports only `--simu gcm`

## 2b. Canonical workflow sequence

The full cleaned `exp5` workflow is not one flat list of interchangeable
phases. The intended sequence is:

1. `prep_phase1`
2. `phase1`
3. `stats`
4. `bc_dataset`
5. `bc_apply`
6. `raw_dataset`
7. `pp_dataset`
8. `train`
9. `predict_loop`
10. `metrics_day`
11. `metrics_month`
12. `value_metrics`
13. `plot_metrics_day`
14. `plot_metrics_month`

In practice:

- `prep_phase1` is optional and only needed if the France-prepared E-OBS target
  files do not already exist
- `raw_dataset` and `pp_dataset` are usually needed for downstream testing and
  evaluation workflows
- `train` is optional if you are reusing an existing checkpoint or checkpoint bundle
- `predict_loop` and the evaluation phases come after a valid trained model or
  reusable checkpoint is available

The practical grouped sequence is:

1. Target preparation:
   - `prep_phase1` if needed
2. Phase 1 training-data build:
   - `phase1`
   - `stats`
3. Phase 2 coarse bias-correction preparation:
   - `bc_dataset`
   - `bc_apply`
4. Test-sample preparation:
   - `raw_dataset`
   - `pp_dataset`
5. Model training or checkpoint reuse:
   - `train` or reuse an existing checkpoint
6. Long prediction:
   - `predict_loop`
7. Evaluation:
   - `metrics_day`
   - `metrics_month`
   - `value_metrics`
   - `plot_metrics_day`
   - `plot_metrics_month`

For the shortest scientifically meaningful historical rerun from an existing
checkpoint, use this sequence:

1. `prep_phase1` if needed
2. `phase1`
3. `stats`
4. `bc_dataset`
5. `bc_apply`
6. `raw_dataset`
7. `pp_dataset`
8. `predict_loop`
9. `metrics_day`
10. `metrics_month`
11. `value_metrics`
12. `plot_metrics_day`
13. `plot_metrics_month`

For the full training recovery sequence, use this order:

1. `prep_phase1` if needed
2. `phase1`
3. `stats`
4. `bc_dataset`
5. `bc_apply`
6. `train`
7. `predict_loop`
8. `metrics_day`
9. `metrics_month`
10. `value_metrics`
11. `plot_metrics_day`
12. `plot_metrics_month`

## 2a. When GPU is actually needed

GPU is mainly useful for:

- `train`
- `predict_loop`

Those are the phases where GPU acceleration matters operationally.

Better GPU candidates:

- `train`: strongly recommended on GPU
- `predict_loop`: recommended on GPU, especially for long periods

Usually fine on CPU:

- `prep_phase1`
- `phase1`
- `stats`
- `bc_dataset`
- `bc_apply`
- `raw_dataset`
- `pp_dataset`
- `metrics_day`
- `metrics_month`
- `value_metrics`
- `plot_metrics_day`
- `plot_metrics_month`

For the other phases, GPU is not scientifically required:

- `prep_phase1`
- `phase1`
- `stats`
- `bc_dataset`
- `bc_apply`
- `raw_dataset`
- `pp_dataset`
- `metrics_day`
- `metrics_month`
- `value_metrics`
- `plot_metrics_day`
- `plot_metrics_month`

Those phases can run on CPU and are the natural candidates for the `globc`
fallback queue when Grace GPU is busy.

Important operational point:

- phases are independent job launches
- you can run early phases on CPU, then switch later phases to GPU
- you do not need to keep the whole workflow on one partition

## 3. Raw-data requirement

By default, the workflow expects raw data under:

```bash
/scratch/globc/page/idownscale_rerun/rawdata
```

That is the main hidden assumption that previously caused confusion.

You have three acceptable setups:

1. real raw files physically inside `repo/rawdata`
2. `rawdata` as a symlink to another location
3. `export IDOWNSCALE_RAW_DIR=/some/other/rawdata`

Recommended quick fix if the data live elsewhere:

```bash
cd /scratch/globc/page/idownscale_rerun
ln -s /path/to/shared/rawdata rawdata
```

Expected top-level layout:

```text
rawdata/
├── eobs/
├── era5/
├── gcm/
└── topography/
```

Raw-data discovery is driven by `iriscc/settings.py`:

- if `IDOWNSCALE_RAW_DIR` is set, that path is used
- otherwise the default is `PROJECT_ROOT/rawdata`

So the engineer does not need to modify code if the raw data are elsewhere.
They only need either:

```bash
export IDOWNSCALE_RAW_DIR=/path/to/rawdata
```

or:

```bash
ln -s /path/to/rawdata /scratch/globc/page/idownscale_rerun/rawdata
```

Important write-path nuance:

- `prep_phase1` writes France-prepared target files into `rawdata/eobs`
- so `prep_phase1` requires `rawdata/eobs` to be writable
- if the engineer has only read access to the shared rawdata tree, they should:
  - use a writable rawdata mirror
  - or skip `prep_phase1` and reuse already-prepared France files

See also `doc/ENVIRONMENT_SETUP.md`.

## 4. Validated Grace GPU environment

Validated working environment:

```bash
/scratch/globc/page/idownscale_envs/production_final_v22_312
```

Validated module stack:

```bash
module load python/gloenv3.12_arm
module load nvidia/cuda/12.4
```

Grace training still benefits from:

```bash
export IDOWNSCALE_FORCE_CSV_LOGGER=1
export IDOWNSCALE_SKIP_TEST_FIGURES=1
```

For preprocessing, statistics, bias correction, metrics, and plotting, GPU is
optional and CPU execution is acceptable.

## 5. Generic Grace GPU submission

For phase orchestration on Grace GPU, use:

```bash
sbatch --export=ALL \
bin/production/submit_obs_workflow_grace.sh
```

That helper defaults to:

- partition: `grace`
- GPU request: `--gres=gpu:1`
- venv: `/scratch/globc/page/idownscale_envs/production_final_v22_312`
- steps: `phase1,stats,bc_dataset,bc_apply`

Use it mainly when:

- you need `train`
- you need `predict_loop`
- or you want one consistent Grace environment for a mixed workflow

Example: if earlier phases already completed on CPU, you can launch only the
next GPU-relevant phase on Grace:

```bash
sbatch --export=ALL,\
STEPS=train,\
TEST_NAME=unet_all,\
IF_EXISTS=overwrite,\
IDOWNSCALE_FORCE_CSV_LOGGER=1,\
IDOWNSCALE_SKIP_TEST_FIGURES=1 \
bin/production/submit_obs_workflow_grace.sh
```

Example: rebuild only Phase 1 and statistics:

```bash
sbatch --export=ALL,\
STEPS=phase1,stats,\
IF_EXISTS=overwrite,\
PHASE1_START_DATE=19800101,\
PHASE1_END_DATE=19800131 \
bin/production/submit_obs_workflow_grace.sh
```

Example: rebuild only coarse bias-correction phases:

```bash
sbatch --export=ALL,\
STEPS=bc_dataset,bc_apply,\
IF_EXISTS=overwrite \
bin/production/submit_obs_workflow_grace.sh
```

Example: long prediction + metrics from an existing checkpoint:

```bash
sbatch --export=ALL,\
STEPS=predict_loop,metrics_day,metrics_month,value_metrics,plot_metrics_day,plot_metrics_month,\
TEST_NAME=unet_all,\
SIMU_TEST=gcm_bc,\
PREDICT_START_DATE=<STARTDATE>,\
PREDICT_END_DATE=<ENDDATE>,\
METRICS_START_DATE=<STARTDATE>,\
METRICS_END_DATE=<ENDDATE>,\
VALUE_START_DATE=<STARTDATE>,\
VALUE_END_DATE=<ENDDATE>,\
IF_EXISTS=overwrite \
bin/production/submit_obs_workflow_grace.sh
```

## 6. Specialized Grace GPU training

For the validated GPU training path, prefer the dedicated submitter:

```bash
sbatch --export=ALL,\
TEST_NAME=unet_smoke,\
STEPS=train,\
IF_EXISTS=overwrite,\
MAX_EPOCH=1,\
IDOWNSCALE_VENV_PATH=/scratch/globc/page/idownscale_envs/production_final_v22_312,\
IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES=tqdm,\
IDOWNSCALE_FORCE_CSV_LOGGER=1,\
IDOWNSCALE_SKIP_TEST_FIGURES=1 \
bin/production/submit_exp5_train_grace.sh
```

That route is the validated one for Grace GPU.

## 7. `globc` CPU fallback

If Grace GPU is full and you want CPU-only workflow phases, use:

```bash
sbatch --export=ALL \
bin/production/submit_obs_workflow_globc.sh
```

This is intended for CPU phases such as:

- `prep_phase1`
- `phase1`
- `stats`
- `bc_dataset`
- `bc_apply`
- `raw_dataset`
- `pp_dataset`
- metrics and plotting steps

This is the preferred fallback when:

- Grace GPU is full
- the engineer only needs preprocessing or evaluation phases
- training is not part of the current rerun

Example: CPU fallback for preprocessing only:

```bash
sbatch --export=ALL,\
STEPS=phase1,stats,\
IF_EXISTS=overwrite \
bin/production/submit_obs_workflow_globc.sh
```

Example: CPU fallback for metrics only:

```bash
sbatch --export=ALL,\
STEPS=metrics_day,metrics_month,value_metrics,plot_metrics_day,plot_metrics_month,\
TEST_NAME=unet_all,\
SIMU_TEST=gcm_bc,\
PREDICT_START_DATE=<STARTDATE>,\
PREDICT_END_DATE=<ENDDATE>,\
METRICS_START_DATE=<STARTDATE>,\
METRICS_END_DATE=<ENDDATE>,\
VALUE_START_DATE=<STARTDATE>,\
VALUE_END_DATE=<ENDDATE>,\
IF_EXISTS=overwrite \
bin/production/submit_obs_workflow_globc.sh
```

Example mixed queue strategy:

1. Start on CPU while GPU is unavailable:

```bash
sbatch --export=ALL,\
STEPS=phase1,stats,bc_dataset,bc_apply,raw_dataset,pp_dataset,\
IF_EXISTS=overwrite \
bin/production/submit_obs_workflow_globc.sh
```

2. Switch to GPU when Grace becomes available:

```bash
sbatch --export=ALL,\
STEPS=train,predict_loop,\
TEST_NAME=unet_all,\
SIMU_TEST=gcm_bc,\
PREDICT_START_DATE=<STARTDATE>,\
PREDICT_END_DATE=<ENDDATE>,\
IF_EXISTS=overwrite,\
IDOWNSCALE_FORCE_CSV_LOGGER=1,\
IDOWNSCALE_SKIP_TEST_FIGURES=1 \
bin/production/submit_obs_workflow_grace.sh
```

3. Move back to CPU for evaluation and plotting:

```bash
sbatch --export=ALL,\
STEPS=metrics_day,metrics_month,value_metrics,plot_metrics_day,plot_metrics_month,\
TEST_NAME=unet_all,\
SIMU_TEST=gcm_bc,\
PREDICT_START_DATE=<STARTDATE>,\
PREDICT_END_DATE=<ENDDATE>,\
METRICS_START_DATE=<STARTDATE>,\
METRICS_END_DATE=<ENDDATE>,\
VALUE_START_DATE=<STARTDATE>,\
VALUE_END_DATE=<ENDDATE>,\
IF_EXISTS=overwrite \
bin/production/submit_obs_workflow_globc.sh
```

Important note:

- the workflow commands are the same on `globc`
- the validated Python environment is the Grace GPU one
- if that venv is not usable on `globc`, provide a partition-appropriate Python
  through `IDOWNSCALE_VENV_PATH` or `IDOWNSCALE_PYTHON_BIN`

## 7a. Avoiding writes to protected directories

An engineer should assume that your personal directories may be read-only for
them unless explicitly shared.

The important write locations are:

- `IDOWNSCALE_OUTPUT_DIR`
- `IDOWNSCALE_REGRID_WEIGHTS_DIR`
- `IDOWNSCALE_DATASET_DIR`
- `IDOWNSCALE_DATASET_BC_DIR`
- `IDOWNSCALE_RUNS_DIR`
- `IDOWNSCALE_PREDICTION_DIR`
- `IDOWNSCALE_METRICS_DIR`
- `IDOWNSCALE_GRAPHS_DIR`
- `IDOWNSCALE_GCM_BC_DIR`
- `IDOWNSCALE_RCM_BC_DIR`

By default:

- outputs go under `/gpfs-calypso/scratch/globc/page/output`
- regrid weights go under `.../output/regrid_weights`
- GCM bias-corrected NetCDFs default under `rawdata/gcm/CNRM-CM6-1-BC`
- RCM bias-corrected NetCDFs default under `rawdata/rcm/ALADIN-BC`

If those locations are not writable for the engineer, they should override them
explicitly before running:

```bash
export IDOWNSCALE_OUTPUT_DIR=/gpfs-calypso/scratch/<user>/output
export IDOWNSCALE_REGRID_WEIGHTS_DIR=$IDOWNSCALE_OUTPUT_DIR/regrid_weights
export IDOWNSCALE_DATASET_DIR=$IDOWNSCALE_OUTPUT_DIR/datasets
export IDOWNSCALE_DATASET_BC_DIR=$IDOWNSCALE_OUTPUT_DIR/datasets/dataset_bc
export IDOWNSCALE_RUNS_DIR=$IDOWNSCALE_OUTPUT_DIR/runs
export IDOWNSCALE_PREDICTION_DIR=$IDOWNSCALE_OUTPUT_DIR/prediction
export IDOWNSCALE_METRICS_DIR=$IDOWNSCALE_OUTPUT_DIR/metrics
export IDOWNSCALE_GRAPHS_DIR=$IDOWNSCALE_GRAPHS_DIR
export IDOWNSCALE_GCM_BC_DIR=$IDOWNSCALE_OUTPUT_DIR/gcm_bc
export IDOWNSCALE_RCM_BC_DIR=$IDOWNSCALE_OUTPUT_DIR/rcm_bc
```

Important nuance:

- `prep_phase1` writes France-prepared E-OBS target files under `rawdata/eobs`
- if the engineer cannot write there, they should either:
  - work against a writable rawdata mirror
  - or reuse already-prepared France target files and skip `prep_phase1`

## 8. Running interactively

For direct shell runs on a Grace-capable environment:

```bash
bash bin/production/run_obs_workflow_grace.sh --exp exp5 --steps phase1,stats
```

The wrapper now checks that raw-data directories exist and fails with a layout
hint if they do not.

## 9. Common phase recipes

Prepare France target files from Europe-scale E-OBS inputs:

```bash
python bin/production/run_obs_workflow.py --exp exp5 --steps prep_phase1
```

Build Phase 1 samples:

```bash
python bin/production/run_obs_workflow.py --exp exp5 --steps phase1
```

Build statistics:

```bash
python bin/production/run_obs_workflow.py --exp exp5 --steps stats
```

Build coarse bias-correction volumes and corrected products:

```bash
python bin/production/run_obs_workflow.py --exp exp5 --steps bc_dataset,bc_apply
```

Build raw GCM test samples:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps raw_dataset \
  --simu gcm
```

Build corrected GCM test samples:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps pp_dataset \
  --simu gcm
```

Train a model:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps train \
  --test-name unet_all
```

Run prediction and evaluation:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps predict_loop,metrics_day,metrics_month,value_metrics,plot_metrics_day,plot_metrics_month \
  --test-name unet_all \
  --simu-test gcm_bc \
  --predict-start-date <STARTDATE> \
  --predict-end-date <ENDDATE> \
  --metrics-start-date <STARTDATE> \
  --metrics-end-date <ENDDATE> \
  --value-start-date <STARTDATE> \
  --value-end-date <ENDDATE>
```

## 10. Environment reproduction

To rebuild the working environment from scratch, start with:

- `doc/ENVIRONMENT_SETUP.md`
- `doc/GRACE_TRAINING_ENGINEER_NOTE.md`

If they only need to verify the existing venv quickly:

```bash
bash bin/production/submit_grace_venv_probe.sh
```

## 10a. Settings the engineer should inspect first

The main settings file is:

```bash
iriscc/settings.py
```

The most useful places to read are:

- path discovery near the top of the file
  - `RAW_DIR`
  - `OUTPUT_DIR`
  - `GCM_BC_DIR`
  - `RCM_BC_DIR`
  - `REGRID_WEIGHTS_DIR`
  - `DATASET_DIR`
  - `RUNS_DIR`
  - `PREDICTION_DIR`
  - `METRICS_DIR`
- the `CONFIG['exp5']` block
  - target grid
  - domain
  - dataset location
  - default SSP/model assumptions

The workflow orchestration logic lives in:

```bash
bin/production/run_obs_workflow.py
```

That is where the engineer should look to understand:

- which phase names exist
- which outputs each phase expects
- which arguments are required for `train`, `predict_loop`, and evaluation

## 11. Operational takeaway

For Calypso reruns, the practical contract is:

1. ensure `rawdata` is visible to the repo
2. use `run_obs_workflow.py` as the master workflow script
3. use `submit_obs_workflow_grace.sh` for Grace GPU orchestration
4. use `submit_obs_workflow_globc.sh` for `globc` CPU fallback
5. use `submit_exp5_train_grace.sh` for validated Grace GPU training
