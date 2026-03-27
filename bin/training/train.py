"""Train the model using PyTorch Lightning.

date: 16/07/2025
author: Zoé GARCIA
"""

import argparse
import sys

sys.path.append('.')

import pathlib

import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger # noqa: E402
# REQUIRED FIX for PyTorch 2.6+ to allow loading checkpoints with Path objects
torch.serialization.add_safe_globals([pathlib.PosixPath]) # noqa: E402

from iriscc.dataloaders import get_dataloaders # noqa: E402
from iriscc.hparams import IRISCCHyperParameters # noqa: E402
from iriscc.lightning_module import IRISCCLightningModule # noqa: E402
from iriscc.lightning_module_ddpm import IRISCCCDDPMLightningModule # noqa: E402
from iriscc.settings import CONFIG # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('--exp', type=str, default='exp5', help='Experiment name')
    args = parser.parse_args()

    # Check if AI step is enabled for this experiment
    if not CONFIG[args.exp].get('ai_step', True):
        print(f"Skipping TRAINING: Experiment {args.exp} is configured to use Bias Correction only.", flush=True)
        sys.exit(0)

    hparams = IRISCCHyperParameters(exp=args.exp)
    train_dataloader = get_dataloaders('train', hparams)
    val_dataloader = get_dataloaders('val', hparams)
    test_dataloader = get_dataloaders('test', hparams)

    if hparams.model == 'cddpm':
        model = IRISCCCDDPMLightningModule(hparams.__dict__)
    else :
        model = IRISCCLightningModule(hparams.__dict__)
    
    logger = TensorBoardLogger(save_dir=hparams.runs_dir, name='lightning_logs')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    torch.set_float32_matmul_precision('high') # For hybrid partition

    trainer = pl.Trainer(max_epochs=hparams.max_epoch, 
                         default_root_dir=hparams.runs_dir,
                         log_every_n_steps=1,
                         accelerator="auto",
                         devices="auto",
                         logger=logger,
                         callbacks=checkpoint_callback)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')