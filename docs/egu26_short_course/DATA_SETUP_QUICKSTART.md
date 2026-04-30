# EGU26 Short Course Data Setup Quickstart

This page gives a simple order of operations for attendees so they are not lost
between Mercure downloads, upstream data retrieval, local directory setup, and the
workflow itself.

## 1. Start from the repository root

All commands below assume you are in the `idownscale` repository root.

## 2. Create the local data tree

Use the helper script:

```bash
bash bin/production/setup_egu26_short_course_tree.sh .
```

This creates the main directories expected by the short-course material, including:

- `rawdata/eobs/`
- `rawdata/era5/tas_1d/`
- `rawdata/gcm/CNRM-CM6-1/`
- `rawdata/gcm/CNRM-CM6-1-BC/`
- `scratch/checkpoint_bundles/`
- `idownscale_output/datasets/dataset_exp5_30y/`
- `idownscale_output/metrics/exp5/mean_metrics/`
- `idownscale_output/graph/metrics/exp5/`
- `idownscale_output/prediction/`

## 3. Download the Mercure material

Mercure URL:

- `https://mercure.cerfacs.fr/egu26scml/`

The easiest starting point is:

- `egu26_sc_required.tar.gz`

Optional additions:

- `egu26_sc_nice_to_have.tar.gz`
- the `raw_data/` folder
- the `phase_outputs/` folder

## 4. Unpack the tar files from the repository root

Example:

```bash
tar -xzf /path/to/egu26_sc_required.tar.gz
tar -xzf /path/to/egu26_sc_nice_to_have.tar.gz
```

These archives unpack into:

- `required/`
- `nice_to_have/`

## 5. Copy the unpacked files into the expected repo locations

### Required tier

```bash
cp -r required/checkpoint_bundles/* scratch/checkpoint_bundles/
cp required/notebook/egu26_short_course_notebook.ipynb docs/egu26_short_course/
cp required/metrics/statistics.json idownscale_output/datasets/dataset_exp5_30y/
cp required/metrics/*.csv idownscale_output/metrics/exp5/mean_metrics/
cp required/plots/*.png idownscale_output/graph/metrics/exp5/
cp required/predictions/*.nc idownscale_output/prediction/
```

### Nice-to-have tier

```bash
cp nice_to_have/eobs_france/*.nc rawdata/eobs/
cp nice_to_have/gcm_bc/*.nc rawdata/gcm/CNRM-CM6-1-BC/
cp -r nice_to_have/checkpoint_bundles/* scratch/checkpoint_bundles/
cp nice_to_have/metrics/*.csv idownscale_output/metrics/exp5/mean_metrics/
cp nice_to_have/plots/*.png idownscale_output/graph/metrics/exp5/
cp nice_to_have/predictions/*.nc idownscale_output/prediction/
```

## 6. Add raw upstream-style files

You have two choices:

### Preferred reproducible route

Fetch ERA5, E-OBS, and CMIP6 from the official upstream repositories, following:

- [How to fetch upstream data](./HOW_TO_FETCH_UPSTREAM_DATA.md)

### Convenience route

Copy the mirrored raw files from Mercure into:

- `rawdata/era5/`
- `rawdata/eobs/`
- `rawdata/gcm/CNRM-CM6-1/`

## 7. Prepare France target files if needed

If the France-specific E-OBS target files are not already present locally:

```bash
python bin/production/run_exp5_workflow.py \
  --exp exp5 \
  --steps prep_phase1
```

Equivalent direct helper:

```bash
python bin/preprocessing/prepare_exp5_france_targets.py
```

## 8. Run the workflow phase by phase

Recommended order:

1. `prep_phase1` (only if needed)
2. `phase1`
3. `stats`
4. optional `train`
5. `predict_loop`
6. `metrics_day`
7. `metrics_month`
8. `value_metrics`
9. `plot_metrics_day`
10. `plot_metrics_month`

See:

- [Local workflow runbook](./LOCAL_WORKFLOW_RUNBOOK.md)
- [Expected outputs by phase](./EXPECTED_PHASE_OUTPUTS.md)

## 9. What to do if you feel lost

Use this decision rule:

- if the file is a **large climate input**, look first in:
  - [How to fetch upstream data](./HOW_TO_FETCH_UPSTREAM_DATA.md)
- if the file is a **project artifact**, look first in:
  - Mercure
- if the file is a **France-prepared target**, run:
  - `prep_phase1`
- if the file is a **workflow output**, check:
  - [Expected outputs by phase](./EXPECTED_PHASE_OUTPUTS.md)

This order usually resolves the confusion quickly.
