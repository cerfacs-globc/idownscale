import sys
sys.path.append('.')

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from iriscc.dataloaders import get_dataloaders
from iriscc.hparams import IRISCCHyperParameters
from iriscc.lightning_module import IRISCCLightningModule
from iriscc.lightning_module_ddpm import IRISCCCDDPMLightningModule

torch.cuda.is_available()

hparams = IRISCCHyperParameters()
train_dataloader = get_dataloaders('train')
val_dataloader = get_dataloaders('val')
test_dataloader = get_dataloaders('test')

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
torch.set_float32_matmul_precision('high')

trainer = pl.Trainer(max_epochs=hparams.max_epoch, 
                     default_root_dir=hparams.runs_dir,
                     log_every_n_steps=1,
                     accelerator="gpu",
                     devices="auto",
                     precision='16-mixed',
                     logger=logger,
                     callbacks=checkpoint_callback)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')