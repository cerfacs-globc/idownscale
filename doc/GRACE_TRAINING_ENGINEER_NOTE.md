# Grace GPU Training Note for `idownscale`

This note records the **working** Grace GPU recipe for training `idownscale` on the
ARM/GH200 nodes. It is meant as an engineer-facing handoff, not polished user
documentation.

## Executive summary

What was finally proven to work on Grace:

- module stack:
  - `python/gloenv3.12_arm`
  - `nvidia/cuda/12.4`
- Python environment:
  - `/scratch/globc/page/idownscale_envs/production_final_v22_312`
- logger mode:
  - force `CSVLogger` instead of TensorBoard during Grace smoke/full training
- training mode:
  - for smoke or production training on Grace, disable post-fit **test figures**
    rather than disabling the whole `test()` path

The key insight is that **the CUDA/runtime compatibility problem was not in our
training code**. It came from using a PyTorch build that expected a newer driver
than the Grace nodes expose. The working route uses a Grace-compatible torch/CUDA
combination.

This note reflects the sequence that was actually proven on Grace:

- a first GPU fit succeeded, but `trainer.test()` failed in figure-generation imports
- a fit-only smoke finished cleanly
- a fit **and** test smoke finished cleanly with `IDOWNSCALE_SKIP_TEST_FIGURES=1`

## What failed

These paths were investigated and are now known bad or misleading:

1. `lib_idownscale_phase2`
   - imports `torch 2.11.0+cu130`
   - Grace accepts the GPU, but the driver/runtime mismatch crashes training

2. local wheel `torch-2.5.1+cu124-cp312-cp312-linux_aarch64.whl`
   - the file in `wheels_production_312/` was not a real wheel
   - it was an `AccessDenied` stub, not a usable artifact

3. mixed system-site plotting stack during test figures
   - training worked
   - post-fit test figures could still crash on `matplotlib/contourpy/GLIBCXX`

## Proven-good environment

The environment path currently used successfully is:

```bash
/scratch/globc/page/idownscale_envs/production_final_v22_312
```

Grace training should run with:

```bash
module load python/gloenv3.12_arm
module load nvidia/cuda/12.4
source /scratch/globc/page/idownscale_envs/production_final_v22_312/bin/activate
```

For safety, also use:

```bash
unset PYTHONHOME
export PYTHONNOUSERSITE=1
```

## Proven-good launch pattern

The Grace submit helper now supports choosing the venv explicitly:

```bash
sbatch --export=ALL,\
TEST_NAME=unet_smoke,\
STEPS=train,\
IF_EXISTS=overwrite,\
IDOWNSCALE_VENV_PATH=/scratch/globc/page/idownscale_envs/production_final_v22_312,\
IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES=tqdm,\
IDOWNSCALE_FORCE_CSV_LOGGER=1,\
IDOWNSCALE_SKIP_TEST_FIGURES=1 \
bin/production/submit_exp5_train_grace.sh
```

Important environment variables:

- `IDOWNSCALE_VENV_PATH`
  - activate this venv before running the workflow
- `IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES`
  - tiny runtime top-up for missing packages such as `tqdm`
- `IDOWNSCALE_FORCE_CSV_LOGGER=1`
  - bypass TensorBoard initialization issues on Grace
- `IDOWNSCALE_SKIP_TEST_FIGURES=1`
  - keep `trainer.test()` from crashing on figure-generation imports
- `IDOWNSCALE_SKIP_TEST=1`
  - optional fallback for the smallest smoke tests, but not needed once the
    figure issue is bypassed

## Reconstruction strategy if the env must be rebuilt

If someone needs to reproduce the Grace-capable env, do **not** start from the
old `lib_idownscale_phase2` overlay.

Safer approach:

1. start from `python/gloenv3.12_arm`
2. load `nvidia/cuda/12.4`
3. create or clone a clean Python 3.12 ARM venv
4. install PyTorch from the PyTorch `cu124` index
5. then install the rest of the Python stack

The important test is not `pip install` success. The important test is:

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

The successful probe looked like:

- `torch 2.5.1`
- `torch.version.cuda == 12.4`
- `torch.cuda.is_available() == True`
- device: `NVIDIA GH200 480GB`

## Current known-good training behavior

What is already proven:

- sanity check passes
- training epoch runs on Grace GPU
- validation runs
- checkpoint is saved
- CSV logs are saved
- post-fit `trainer.test()` also runs if test figures are disabled

The smoke run produced outputs under:

```bash
/gpfs-calypso/scratch/globc/page/idownscale_output/runs/exp5/unet_smoke/lightning_logs/version_best/
```

Including:

- `metrics.csv`
- `hparams.yaml`
- `checkpoints/best-checkpoint-epoch=00-val_loss=...ckpt`
- `metrics_test_set.csv` when the test phase is allowed to complete

## Recommended run modes

### Smallest smoke test

Use this when you only want to prove that GPU fit works:

```bash
sbatch --export=ALL,\
TEST_NAME=unet_smoke,\
STEPS=train,\
IF_EXISTS=overwrite,\
MAX_EPOCH=1,\
IDOWNSCALE_VENV_PATH=/scratch/globc/page/idownscale_envs/production_final_v22_312,\
IDOWNSCALE_VENV_BOOTSTRAP_PACKAGES=tqdm,\
IDOWNSCALE_FORCE_CSV_LOGGER=1,\
IDOWNSCALE_SKIP_TEST=1 \
bin/production/submit_exp5_train_grace.sh
```

### Preferred Grace validation smoke

Use this when you want fit **and** test metrics:

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

This is the mode that currently gives the best balance between:

- genuine end-to-end training validation
- test metrics production
- minimal exposure to Grace plotting-stack fragility

## Remaining caution

The Grace training path is now operational, but post-fit evaluation/plotting
should be treated separately from the core training environment.

In particular:

- Grace smoke/full training:
  - use CSV logger
  - disable test figures
- richer evaluation/visualization:
  - run in a better-controlled analysis environment if plots are required

That separation is simpler and more robust than trying to make one environment
do everything at once.

## Related files in this repo

- Grace submitter:
  - `bin/production/submit_exp5_train_grace.sh`
- training entrypoint:
  - `bin/training/train.py`
- Lightning modules:
  - `iriscc/lightning_module.py`
  - `iriscc/lightning_module_ddpm.py`

## Operational takeaway

The practical contract for engineers is:

- use the Grace-compatible venv
- use CUDA 12.4
- force CSV logging on Grace
- do not rely on TensorBoard or test figures for the first smoke runs
- treat training and rich evaluation as two different runtime layers
