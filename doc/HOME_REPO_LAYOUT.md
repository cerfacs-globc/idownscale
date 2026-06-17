# Home Repository Layout

This note documents the recommended layout when the Git repository lives in
backed-up `home` while heavy runtime artifacts live on `scratch`.

## Recommended split

Use:

```text
/home/globc/<login>/src/idownscale
```

for the repository itself, and:

```text
/scratch/globc/<login>/idownscale_runtime/
├── rawdata/
├── output/
│   ├── datasets/
│   ├── runs/
│   ├── prediction/
│   ├── metrics/
│   └── regrid_weights/
├── graphs/
└── tmp/
```

for generated data, runs, metrics, figures, regridding weights, and transient
runtime files.

## Why this split

- the repository and `.git` metadata stay on the more stable, backed-up
  filesystem
- large NetCDF/sample/checkpoint traffic stays on scratch
- a scratch I/O issue is less likely to damage the Git repository itself
- cleanup is simpler because runtime artifacts are outside the code tree

## Environment variables

The minimal setup is:

```bash
export IDOWNSCALE_RUNTIME_ROOT=/scratch/globc/$USER/idownscale_runtime
export IDOWNSCALE_RAW_DIR=$IDOWNSCALE_RUNTIME_ROOT/rawdata
export IDOWNSCALE_OUTPUT_DIR=$IDOWNSCALE_RUNTIME_ROOT/output
export IDOWNSCALE_REGRID_WEIGHTS_DIR=$IDOWNSCALE_OUTPUT_DIR/regrid_weights
export IDOWNSCALE_DATASET_DIR=$IDOWNSCALE_OUTPUT_DIR/datasets
export IDOWNSCALE_RUNS_DIR=$IDOWNSCALE_OUTPUT_DIR/runs
export IDOWNSCALE_PREDICTION_DIR=$IDOWNSCALE_OUTPUT_DIR/prediction
export IDOWNSCALE_METRICS_DIR=$IDOWNSCALE_OUTPUT_DIR/metrics
export IDOWNSCALE_GRAPHS_DIR=$IDOWNSCALE_RUNTIME_ROOT/graphs
```

Optional:

```bash
export TMPDIR=$IDOWNSCALE_RUNTIME_ROOT/tmp
mkdir -p "$TMPDIR"
```

## Current default behavior

The Python settings layer now behaves like this:

1. `RAW_DIR = IDOWNSCALE_RAW_DIR` if set
2. otherwise `RAW_DIR = repo/rawdata` if that directory exists
3. otherwise `RAW_DIR = IDOWNSCALE_RUNTIME_ROOT/rawdata`

and:

1. `OUTPUT_DIR = IDOWNSCALE_OUTPUT_DIR` if set
2. otherwise `OUTPUT_DIR = IDOWNSCALE_RUNTIME_ROOT/output`

This means a repository in `home` will not silently create a large
`repo/output` tree anymore unless you explicitly point it there.

## Shell wrappers

The main workflow wrappers were aligned with the same logic:

- `bin/production/run_in_grace_env.sh`
- `bin/production/run_obs_workflow_grace.sh`
- `bin/production/submit_obs_workflow_grace.sh`
- `bin/production/submit_obs_workflow_globc.sh`
- `bin/production/submit_exp5_perfect_model_grace.sh`
- `bin/production/submit_exp5_perfect_model_kraken.sh`

They now default to a per-user scratch runtime root instead of assuming
`repo/rawdata` and a project-owned output tree.

## Important caveat

Some preprocessing phases still write derived products under `rawdata/` on
purpose, for example target preparation or some bias-correction side products.
So `rawdata/` must remain writable for those workflows, but it does not need to
live inside the repository anymore.
