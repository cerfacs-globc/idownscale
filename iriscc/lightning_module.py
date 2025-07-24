"""
Lightning module for IRISCC project.

date : 16/07/2025
Inspired by Météo-France segmentation Lightning Module https://github.com/meteofrance/mfai/tree/main
Modified by Zoé GARCIA
"""

import sys
sys.path.append('.')

from pathlib import Path
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR

from iriscc.metrics import MaskedMAE, MaskedRMSE
from iriscc.models.unet import UNet
from iriscc.models.miniunet import MiniUNet
from iriscc.models.miniswinunetr import MiniSwinUNETR
from iriscc.loss import MaskedMSELoss, MaskedGammaMAELoss

layout = {
    "Check Overfit": {
        "loss": ["Multiline", ["loss/train", "loss/val"]],
    },
}

def get_model(model:str, 
              in_channels:int, 
              out_channels:int, 
              img_size:tuple):
    
    match model:
        case 'unet':
            return UNet(in_channels=in_channels, out_channels=out_channels, init_features=32).float()
        case 'miniunet':
            return MiniUNet(in_channels=in_channels, out_channels=out_channels, init_features=32).float()
        case 'swinunetr':
            return SwinUNETR(img_size=img_size, in_channels=in_channels, out_channels=out_channels, spatial_dims=2)
        case 'miniswinunetr':
            return MiniSwinUNETR(img_size=img_size, in_channels=in_channels, out_channels=out_channels, spatial_dims=2)
        


class IRISCCLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
 
        self.fill_value = hparams['fill_value']    
        self.learning_rate = hparams['learning_rate']
        self.runs_dir = hparams['runs_dir']
        self.scheduler_step_size = hparams['scheduler_step_size']
        self.scheduler_gamma = hparams['scheduler_gamma']
        self.in_channels = hparams['in_channels']
        self.img_size = hparams['img_size']
        os.makedirs(self.runs_dir, exist_ok=True)

        loss_name = hparams['loss']
        if loss_name == 'masked_gamma_mae':
            self.loss = MaskedGammaMAELoss(ignore_value=self.fill_value,
                                           sample_dir=hparams['sample_dir'])
        else :
            self.loss = MaskedMSELoss(ignore_value=self.fill_value)
        self.metrics_dict = nn.ModuleDict({
                    "rmse": MaskedRMSE(ignore_value = self.fill_value),
                    "mae": MaskedMAE(ignore_value = self.fill_value)
                })
        self.model = get_model(model=hparams['model'], 
                               in_channels=self.in_channels, 
                               out_channels=1, 
                               img_size=self.img_size)

        self.test_metrics = {}
        self.train_step_outputs = []
        self.val_step_outputs = []
        
        self.save_hyperparameters()
        self.epoch_start_time = None

    def forward(self, x):
        return self.model(x) 

    def on_train_start(self):
        self.logger.experiment.add_custom_scalars(layout)
        self.logger.log_hyperparams(vars(self.hparams))

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def common_step(self, x, y):
        y_hat = self(x)
        loss = torch.sqrt(self.loss(y_hat, y))
        return y_hat, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self.common_step(x, y)
        self.train_step_outputs.append(loss)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.train_step_outputs).mean()
        self.logger.experiment.add_scalar("loss/train", epoch_average, self.current_epoch)
        self.train_step_outputs.clear()
        epoch_duration = time.time() - self.epoch_start_time
        self.log("epoch_time", epoch_duration, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self.common_step(x, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.val_step_outputs).mean()
        self.logger.experiment.add_scalar("loss/val", epoch_average, self.current_epoch)
        self.val_step_outputs.clear()

        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self.common_step(x, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            
        batch_dict = {"loss": loss}
        for metric_name, metric in self.metrics_dict.items():
            metric.update(y_hat, y)
            batch_dict[metric_name] = metric.compute()
            self.logger.experiment.add_scalar(metric_name, metric.compute(), batch_idx)
            metric.reset()
        self.test_metrics[batch_idx] = batch_dict

        if batch_idx == 0:

            fig, ax = plt.subplots()
            y[y == self.fill_value] = torch.nan
            vmin, vmax = np.nanmin(y.cpu().numpy()), np.nanmax(y.cpu().numpy())
            levels = np.round(np.linspace(vmin, vmax, 11)).astype(int)
            cs = ax.contourf(y[batch_idx,0,:,:].cpu().numpy(), cmap='OrRd', levels=levels)
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_y_0', fig)
    
            fig, ax = plt.subplots()
            x[x == self.fill_value] = torch.nan
            cs = ax.contourf(x[batch_idx,-1,:,:].cpu().numpy(), cmap='OrRd')
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_x_0', fig)

            fig, ax = plt.subplots()
            cs = ax.contourf(y_hat[batch_idx,0,:,:].cpu().numpy(), cmap='OrRd')
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_yhat_raw_0', fig)

            fig, ax = plt.subplots()
            y_hat[torch.isnan(y)] = torch.nan 
            cs = ax.contourf(y_hat[batch_idx,0,:,:].cpu().numpy(), cmap='OrRd', levels=levels)
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_yhat_0', fig)
 
            
    def build_metrics_dataframe(self):
        data = []
        first_sample = list(self.test_metrics.keys())[0]
        metrics = list(self.test_metrics[first_sample].keys())
        for name_sample, metrics_dict in self.test_metrics.items():
            data.append([name_sample] + [metrics_dict[m].item() for m in metrics])
        return pd.DataFrame(data, columns=["Name"] + metrics)

    def save_test_metrics_as_csv(self, df):
        path_csv = Path(self.logger.log_dir) / "metrics_test_set.csv"
        df.to_csv(path_csv, index=False)
    
    def on_test_epoch_end(self):
        df = self.build_metrics_dataframe()
        self.save_test_metrics_as_csv(df)
        df = df.drop("Name", axis=1)
        self.log('hp_metric', df['rmse'].mean())
    
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        return optimizer


