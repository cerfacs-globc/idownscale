# EGU26 Short Course Environment Setup

> Release compatibility: this EGU26 short-course material is maintained against `idownscale` release `v1.4.0`. If you are using another release, check that workflow runner names, paths, and expected outputs still match that version.

This page gives the minimum environment setup needed to run the short-course
workflow locally.

For the fuller project documentation, see:

- `docs/getting_started.rst`
- `docs/training.rst`

## 0. Check out the course-compatible release

The maintained short-course material matches `idownscale` release `v1.4.0`.
For the cleanest student setup:

```bash
git clone https://github.com/cerfacs-globc/idownscale.git
cd idownscale
git checkout v1.4.0
```

If you use a newer branch or `master`, some commands, file names, or workflow
details may differ from the teaching material.

## 1. Create the Python environment

The recommended route is Conda.

At the moment, this repository does **not** provide a ready-to-use
`environment.yml`, so the environment must be created explicitly.

```bash
conda create -n idownscale -c conda-forge \
  python=3.11 \
  esmpy xesmf netcdf4 cartopy \
  numpy scipy pandas xarray \
  matplotlib seaborn pyproj
conda activate idownscale
pip install -r requirements.txt
pip install -e .
```

This route is intentionally close to the working installation logic already used in
the repository.

## 2. Set the main runtime paths

For a self-contained local clone, from the repository root:

```bash
export IDOWNSCALE_RAW_DIR=$PWD/rawdata
export IDOWNSCALE_OUTPUT_DIR=$PWD/output
export IDOWNSCALE_GRAPHS_DIR=$PWD/graphs
export IDOWNSCALE_REGRID_WEIGHTS_DIR=$PWD/output/regrid_weights
export IDOWNSCALE_RUNS_DIR=$PWD/output/runs
export IDOWNSCALE_PREDICTION_DIR=$PWD/output/prediction
export IDOWNSCALE_METRICS_DIR=$PWD/output/metrics
```

These settings keep the short-course workflow self-contained inside the local
clone.

If you use the repository on an HPC system and want the code in backed-up
`home`, prefer:

```bash
export IDOWNSCALE_RUNTIME_ROOT=/scratch/globc/$USER/idownscale_runtime
export IDOWNSCALE_RAW_DIR=$IDOWNSCALE_RUNTIME_ROOT/rawdata
export IDOWNSCALE_OUTPUT_DIR=$IDOWNSCALE_RUNTIME_ROOT/output
export IDOWNSCALE_GRAPHS_DIR=$IDOWNSCALE_RUNTIME_ROOT/graphs
export IDOWNSCALE_REGRID_WEIGHTS_DIR=$IDOWNSCALE_OUTPUT_DIR/regrid_weights
export IDOWNSCALE_RUNS_DIR=$IDOWNSCALE_OUTPUT_DIR/runs
export IDOWNSCALE_PREDICTION_DIR=$IDOWNSCALE_OUTPUT_DIR/prediction
export IDOWNSCALE_METRICS_DIR=$IDOWNSCALE_OUTPUT_DIR/metrics
```

## 3. Create the local directory tree

```bash
bash bin/production/setup_egu26_short_course_tree.sh .
```

## 4. Add the course data

Use either:

- Mercure tar files and folders
- or the official upstream sources, following:
  - [How to fetch upstream data](./HOW_TO_FETCH_UPSTREAM_DATA.md)

For the recommended order of operations, see:

- [Data setup quickstart](./DATA_SETUP_QUICKSTART.md)

## 5. First workflow check

Once the environment and data are in place, the first lightweight command is:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps prep_phase1
```

If the France-specific E-OBS target files already exist locally, you can skip that
step and continue with:

```bash
python bin/production/run_obs_workflow.py \
  --exp exp5 \
  --steps phase1,stats \
  --test-name unet_course_demo
```

## 6. Notes

- Long-running phases are normal for this workflow.
- Training is optional for the short course.
- The notebook is designed as a guided example, not as a promise that every phase is
  fast on a laptop.

## 7. Usual technical problems

### `xesmf` / `ESMF` / `esmpy`

This is the most common installation issue.

- `xesmf` depends on `ESMF` / `esmpy`
- installing `xesmf` with `pip` alone is often not enough
- this is why the recommended route installs `esmpy` and `xesmf` through Conda first

If `import xesmf` fails, recreate the environment with the Conda-based route above.

## 8. What may differ from later code versions

The short course focuses on the E-OBS/GCM daily temperature example and does
not cover the full breadth of `v1.4.0`. Later or broader workflows in the
release include:

- the generic observation-target runner name `run_obs_workflow.py`
- CERRA-based observation-target work
- fixed-step mixed prediction cadences
- perfect-model BC+CDDPM workflows
- richer provenance inventories

Those features are available in the release, but they are beyond the minimum
student path described here.

HPC note:

- on some HPC systems, `xesmf` may also need an environment variable such as
  `ESMFMKFILE` so that `ESMF` / `esmpy` can be discovered correctly
- the exact value is installation-specific and should come from the local
  platform documentation or support team

Typical pattern:

```bash
export ESMFMKFILE=/path/to/esmf.mk
```

### `SBCK`

`SBCK` is listed in `requirements.txt`:

- `SBCK==1.4.2`

It is used for bias-correction-related tooling. If `pip install -r requirements.txt`
fails on `SBCK`, first make sure the Conda base environment is active and the standard
scientific stack is already installed.

For the short course, this package matters mainly if users want to reproduce the
bias-correction side of the workflow, not only inference.

### `cartopy`

`cartopy` can fail if installed only through `pip`, especially when system geospatial
dependencies are incomplete.

Recommended route:

- install `cartopy` through Conda, as shown above

### `torch` / `pytorch_lightning`

For local CPU-only use, the standard `pip install -r requirements.txt` route may be
enough.

For GPU/HPC use, the exact `torch` environment can matter a lot. The project docs
contain internal HPC-oriented notes, but for the short course it is better to
present local CPU use as the default.

### TensorBoard not available

If TensorBoard is not installed or not working, training can still proceed.

The repository supports CSV logging through:

```bash
export IDOWNSCALE_FORCE_CSV_LOGGER=1
```

This is especially useful on constrained or unusual environments.

### Plotting failures after training

Some environments can fail during post-training figure generation because of plotting
stack issues.

If needed, figures can be skipped during training-time testing with:

```bash
export IDOWNSCALE_SKIP_TEST_FIGURES=1
```

### If installation still feels fragile

Use this fallback order:

1. create a fresh Conda environment
2. install the geospatial and remapping stack first:
   - `esmpy`, `xesmf`, `netcdf4`, `cartopy`
3. install `requirements.txt`
4. install the project with `pip install -e .`

That usually resolves the most common issues faster than trying to patch a broken
environment incrementally.
