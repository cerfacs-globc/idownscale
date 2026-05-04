# Environment Setup For The EGU26 Short Course

This page describes the recommended Python environment setup for the EGU26 short
course and the main issues already encountered during development.

The short version is:

- prefer Conda or Mamba for the base environment
- install `esmpy` and `xesmf` from Conda
- set `ESMFMKFILE` correctly
- install some packages with `pip` only after the Conda base is stable
- verify imports before starting the workflow

## 1. Why Conda is recommended

This project depends on `xesmf`, which in turn depends on `ESMF` and `esmpy`.
That stack is much easier to install reliably with Conda than with a pure `pip`
environment.

Recommended base packages from Conda:

- `python`
- `eigen`
- `esmpy`
- `xesmf`
- `netcdf4`
- `cartopy`
- `numpy`
- `scipy`
- `pandas`
- `xarray`
- `matplotlib`
- `seaborn`
- `pyproj`
- `pytorch`
- `torchvision`
- `torchaudio`
- `pytorch-lightning`
- `torchmetrics`
- `timm`
- `tqdm`

## 2. Known practical issues

### ESMF / esmpy / xesmf

This is the most fragile part of the environment.

Known points:

- `xesmf` may fail if `esmpy` is missing or mismatched
- on some systems, `ESMFMKFILE` must be set explicitly
- ARM and HPC systems can be more sensitive to package mixing

The validated `setup_env.sh` route exports:

```bash
export ESMFMKFILE="$CONDA_PREFIX/lib/esmf.mk"
```

If `xesmf` import or regridding fails, checking `ESMFMKFILE` should be one of
the first debugging steps.

### Mixed Conda and system Python

On HPC systems, loaded modules can inject `PYTHONHOME` or `PYTHONPATH` in ways
that break the Conda environment.

The validated HPC setup explicitly does:

```bash
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
```

This is important when a module-loaded Python would otherwise leak system
packages into the environment.

### SBCK

`SBCK` is not the main validated path for the current short course, but it is a
known dependency that may be useful in some environments or experiments.

The environment script installs it with `pip`:

```bash
pip install SBCK==1.4.2
```

If `SBCK` installation fails, that should not block the main `ibicus`-based
workflow unless the notebook explicitly chooses to demonstrate `SBCK`.

### ibicus

The validated bias-correction path uses `ibicus`, which is installed with `pip`:

```bash
pip install ibicus==1.1.1
```

Users should verify that `ibicus` imports cleanly before starting the bias-correction phases.

## 3. Recommended installation sequence

### Local workstation or laptop

Use Conda or Mamba to create the base environment first, then install the
remaining packages.

Example sequence:

```bash
conda create -n idownscale -c conda-forge python=3.11
conda activate idownscale
conda install -c conda-forge \
  eigen esmpy xesmf netcdf4 cartopy \
  numpy scipy pandas xarray \
  matplotlib seaborn pyproj \
  pytorch torchvision torchaudio pytorch-lightning torchmetrics \
  timm tqdm
export ESMFMKFILE="$CONDA_PREFIX/lib/esmf.mk"
pip install ibicus==1.1.1 monai==1.4.0 SBCK==1.4.2
pip install -e .
```

### Grace or similar HPC system

The repo already contains a validated HPC setup script:

- [setup_env.sh](/Users/page/src/idownscale/setup_env.sh)

That script is useful mainly as a reference because it documents the exact
environment workarounds that were needed on the target platform.

Important details from that script:

- Conda-based environment creation
- `esmpy` and `xesmf` installed from Conda
- `ESMFMKFILE` exported explicitly
- `PYTHONHOME` and `PYTHONPATH` unset before `pip`
- `SBCK`, `ibicus`, and `monai` installed with `pip`

## 4. Minimal verification before starting the notebook

Before starting the workflow, users should check that these imports work:

```bash
python -c "import xesmf, ibicus, cartopy, torch; print('ok')"
```

A stronger check is:

```bash
python -c "import xesmf, ibicus, cartopy, torch; print(torch.__version__); print(xesmf.__version__)"
```

If this import check fails, the workflow should not be started yet.

## 5. Minimal smoke tests

After the import check, users should verify that the local CLI entrypoints work.

No-data smoke tests:

```bash
bash bin/production/setup_egu26_short_course_tree.sh .
python bin/preprocessing/crop_domain.py --help
python bin/production/run_exp5_workflow.py --help
```

If local data are already in place, a stronger practical smoke test is a very
short Phase 1 run such as:

```bash
python bin/production/run_exp5_workflow.py \
  --exp exp5 \
  --steps phase1 \
  --phase1-start-date 19850101 \
  --phase1-end-date 19850103
```

## 6. What the notebook should explain

The notebook should include a short environment section near the beginning:

- why Conda is recommended
- why `xesmf` and `ESMF` are the main installation pain points
- why `ESMFMKFILE` may need to be set manually
- which packages are installed with Conda and which are installed with `pip`
- how to run a short import verification before moving to the data phases

This section should be explanatory, but also practical enough that users can
copy the commands and get to a working environment.

Right after that, the notebook should mention the Mercure `.tar.gz` files as an
easy way to retrieve the published course assets.
