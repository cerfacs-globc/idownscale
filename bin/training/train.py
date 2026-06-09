"""
Train a model using PyTorch Lightning.

date: 16/07/2025
author: Zoé GARCIA
"""

import argparse
import os
import shutil
import sys
sys.path.append('.')

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.loggers import TensorBoardLogger
except ModuleNotFoundError:
    TensorBoardLogger = None

from iriscc.dataloaders import get_dataloaders
from iriscc.hparams import IRISCCHyperParameters
from iriscc.lightning_module import IRISCCLightningModule
from iriscc.lightning_module_ddpm import IRISCCCDDPMLightningModule


def prepare_normalization_statistics(hparams: IRISCCHyperParameters) -> None:
    source = hparams.sample_dir / "statistics.json"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing training statistics file: {source}. "
            "Run bin/preprocessing/compute_statistics.py on the exact training sample directory before training."
        )
    stats_dir = hparams.runs_dir / "normalization_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    target = stats_dir / "statistics.json"
    shutil.copy2(source, target)
    hparams.statistics_dir = stats_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IRISCC model.")
    parser.add_argument("--exp", default="exp5", help="Experiment name, e.g. exp5.")
    parser.add_argument("--test-name", default="unet_all", help="Run name used under runs/<exp>/.")
    parser.add_argument("--model", default="unet", help="Model family, e.g. unet or cddpm.")
    parser.add_argument("--max-epoch", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=8e-4, help="Optimizer learning rate.")
    parser.add_argument(
        "--loss",
        default=None,
        help="Override loss name. Defaults to masked_mse for temperature and masked_gamma_mae for precipitation.",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for supported architectures.")
    parser.add_argument("--output-norm", action="store_true", help="Enable output normalization in transforms.")
    parser.add_argument("--skip-test", action="store_true", help="Skip the post-fit test pass.")
    parser.add_argument("--sample-dir", default=None, help="Optional override for the training sample directory.")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducibility seed for training.")
    parser.add_argument("--n-steps", type=int, default=200, help="CDDPM diffusion steps.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision('high')
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
    if args.model == "cddpm" and not args.output_norm:
        raise ValueError("CDDPM training requires --output-norm so the diffusion target is sampled in normalized space.")

    hparams = IRISCCHyperParameters(
        exp=args.exp,
        run_name=args.test_name,
        model=args.model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        loss=args.loss,
        dropout=args.dropout,
        output_norm=args.output_norm,
        sample_dir=args.sample_dir,
        seed=args.seed,
        n_steps=args.n_steps,
        output_range="minus_one_one" if args.model == "cddpm" else "zero_one",
    )
    prepare_normalization_statistics(hparams)

    train_dataloader = get_dataloaders('train', hparams)
    val_dataloader = get_dataloaders('val', hparams)
    test_dataloader = get_dataloaders('test', hparams)

    if hparams.model == 'cddpm':
        model = IRISCCCDDPMLightningModule(hparams.__dict__)
    else:
        model = IRISCCLightningModule(hparams.__dict__)

    force_csv_logger = os.getenv("IDOWNSCALE_FORCE_CSV_LOGGER", "").lower() in {"1", "true", "yes", "on"}
    skip_test = args.skip_test or os.getenv("IDOWNSCALE_SKIP_TEST", "").lower() in {"1", "true", "yes", "on"}

    if TensorBoardLogger is not None and not force_csv_logger:
        try:
            logger = TensorBoardLogger(
                save_dir=hparams.runs_dir,
                name='lightning_logs',
                version='version_best',
            )
        except ModuleNotFoundError:
            logger = CSVLogger(
                save_dir=hparams.runs_dir,
                name='lightning_logs',
                version='version_best',
            )
    else:
        logger = CSVLogger(
            save_dir=hparams.runs_dir,
            name='lightning_logs',
            version='version_best',
        )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
        precision = '16-mixed'
    else:
        accelerator = "cpu"
        precision = '32-true'

    trainer = pl.Trainer(
        max_epochs=hparams.max_epoch,
        default_root_dir=hparams.runs_dir,
        log_every_n_steps=1,
        accelerator=accelerator,
        devices="auto",
        precision=precision,
        logger=logger,
        callbacks=checkpoint_callback,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    if not skip_test:
        trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
