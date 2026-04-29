# Grace training validation note

This note captures the first successful GPU training validation on Grace for the
current `idownscale` branch.

## Successful runs

- `unet_smoke`
  - 1 epoch
  - Grace GPU fit + test completed successfully
  - used:
    - `IDOWNSCALE_FORCE_CSV_LOGGER=1`
    - `IDOWNSCALE_SKIP_TEST_FIGURES=1`

- `unet_grace30`
  - 30 epochs
  - Grace GPU fit + test completed successfully
  - best checkpoint:
    - `best-checkpoint-epoch=26-val_loss=0.77.ckpt`

## Output locations

Smoke run:

```bash
/gpfs-calypso/scratch/globc/page/idownscale_output/runs/exp5/unet_smoke/lightning_logs/version_best/
```

30-epoch run:

```bash
/gpfs-calypso/scratch/globc/page/idownscale_output/runs/exp5/unet_grace30/lightning_logs/version_best/
```

Important files:

- `metrics.csv`
- `metrics_test_set.csv`
- `hparams.yaml`
- `checkpoints/best-checkpoint-*.ckpt`

## Archive comparison

Reference archive bundle:

```bash
/gpfs-calypso/scratch/globc/page/idownscale_rerun/scratch/checkpoint_bundles/exp5_unet_all_bundle/
```

Archive checkpoint:

- `best-checkpoint-epoch=59-val_loss=0.76.ckpt`

Archive mean test metrics:

- `loss = 0.732006`
- `rmse = 0.732006`
- `mae  = 0.567759`

Grace 30-epoch run mean test metrics:

- `loss = 0.742423`
- `rmse = 0.742423`
- `mae  = 0.573866`

Delta (`current - archive`):

- `loss = +0.010416`
- `rmse = +0.010416`
- `mae  = +0.006107`

Interpretation:

- this is close enough to count as a meaningful archival recovery of the
  training path on Grace GPU
- the result is not bitwise identical to the original archive, but it is in the
  same performance neighborhood
- the recovered training workflow is operational and scientifically credible

## Plots

Generated training-history plots:

- `scratch/training_plots/exp5_unet_smoke_history_posttest.png`
- `scratch/training_plots/exp5_unet_grace30_history_final.png`
- `scratch/training_plots/exp5_test_metric_means_comparison.png`

These plots summarize:

- train/validation loss vs step
- train/validation loss vs epoch
- epoch duration
- archive vs smoke vs Grace-30 mean test metrics
