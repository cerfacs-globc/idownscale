# EGU26 Short Course Data Setup Quickstart

This page gives the shortest reliable path for users who start from the GitHub
materials after the course and want a working local layout quickly.

## 1. Work from the repository root

All commands below assume you are in the `idownscale` repository root.

## 2. Create the directory tree

Run:

```bash
bash bin/production/setup_egu26_short_course_tree.sh .
```

This creates the main runtime directories used by the short-course material:

- `rawdata/eobs/`
- `rawdata/era5/tas_1d/`
- `rawdata/gcm/CNRM-CM6-1/`
- `rawdata/gcm/CNRM-CM6-1-BC/`
- `scratch/checkpoint_bundles/`
- `idownscale_output/datasets/dataset_exp5_30y/`
- `idownscale_output/metrics/exp5/mean_metrics/`
- `idownscale_output/graph/metrics/exp5/`
- `idownscale_output/prediction/`
- `idownscale_output/runs/exp5/`

## 3. Download the published course material

Mercure root:

- `https://mercure.cerfacs.fr/egu26scml/`

The easiest starting point is:

- `egu26_sc_required.tar.gz`

Optional additions:

- `egu26_sc_nice_to_have.tar.gz`
- `raw_data/`
- `phase_outputs/`

## 4. Unpack from the repository root

Example:

```bash
tar -xzf /path/to/egu26_sc_required.tar.gz
tar -xzf /path/to/egu26_sc_nice_to_have.tar.gz
```

These archives unpack into:

- `required/`
- `nice_to_have/`

## 5. Copy the published files into the local runtime tree

Required tier:

```bash
cp -r required/checkpoint_bundles/* scratch/checkpoint_bundles/
cp required/metrics/statistics.json idownscale_output/datasets/dataset_exp5_30y/
cp required/metrics/*.csv idownscale_output/metrics/exp5/mean_metrics/
cp required/plots/*.png idownscale_output/graph/metrics/exp5/
cp required/predictions/*.nc idownscale_output/prediction/
```

Optional nice-to-have tier:

```bash
cp nice_to_have/eobs_france/*.nc rawdata/eobs/
cp nice_to_have/gcm_bc/*.nc rawdata/gcm/CNRM-CM6-1-BC/
cp -r nice_to_have/checkpoint_bundles/* scratch/checkpoint_bundles/
cp nice_to_have/metrics/*.csv idownscale_output/metrics/exp5/mean_metrics/
cp nice_to_have/plots/*.png idownscale_output/graph/metrics/exp5/
cp nice_to_have/predictions/*.nc idownscale_output/prediction/
```

## 6. Add raw climate inputs

Two routes are possible.

Preferred reproducible route:

- fetch ERA5, E-OBS, and CMIP6 from the official upstream sources
- follow [HOW_TO_FETCH_UPSTREAM_DATA.md](./HOW_TO_FETCH_UPSTREAM_DATA.md)

Convenience route:

- copy the mirrored files from Mercure into `rawdata/era5/`, `rawdata/eobs/`,
  and `rawdata/gcm/CNRM-CM6-1/`

## 7. Prepare the France target files if needed

If the France-cropped E-OBS files are not already present locally, use the
existing crop helper from this repo.

Temperature target:

```bash
python bin/preprocessing/crop_domain.py \
  --input rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231.nc \
  --output rawdata/eobs/tas_ens_mean_1d_025deg_reg_v29_0e_19500101-20231231_france.nc \
  --exp exp5 \
  --standardize
```

Elevation target:

```bash
python bin/preprocessing/crop_domain.py \
  --input rawdata/eobs/elevation_ens_025deg_reg_v29_0e.nc \
  --output rawdata/eobs/elevation_ens_025deg_reg_v29_0e_france.nc \
  --exp exp5 \
  --standardize
```

Users who only want to inspect later phases can also reuse the already published
France-cropped files from `nice_to_have/eobs_france/`.

## 8. Run the workflow phase by phase

Recommended order:

1. `phase1`
2. `stats`
3. `bc_dataset`
4. `bc_apply`
5. optional `train`
6. `predict_loop`
7. `metrics_day`
8. `metrics_month`
9. `value_metrics`
10. `plot_metrics_day`
11. `plot_metrics_month`

See:

- [LOCAL_WORKFLOW_RUNBOOK.md](./LOCAL_WORKFLOW_RUNBOOK.md)
- [EXPECTED_PHASE_OUTPUTS.md](./EXPECTED_PHASE_OUTPUTS.md)

## 9. Short decision rule

- if the file is a large climate input, look first in
  [HOW_TO_FETCH_UPSTREAM_DATA.md](./HOW_TO_FETCH_UPSTREAM_DATA.md)
- if the file is a published project artifact, look first in Mercure
- if the file is a France-cropped E-OBS target, either reuse
  `nice_to_have/eobs_france/` or run `crop_domain.py`
- if the file is a workflow output, check
  [EXPECTED_PHASE_OUTPUTS.md](./EXPECTED_PHASE_OUTPUTS.md)
