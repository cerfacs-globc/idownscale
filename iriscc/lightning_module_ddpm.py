"""
Lightning module for the IRISCC CDDPM model.

date : 16/07/2025
Rachid Elmontassir script modified by Zoé Garcia
"""

import sys
sys.path.append('.')

from pathlib import Path
import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from iriscc.transforms import DeMinMaxNormalisation
from iriscc.metrics import MaskedMAE, MaskedRMSE
from iriscc.models.cddpm import CDDPM
from iriscc.loss import MaskedMSELoss

layout = {
    "Check Overfit": {
        "loss": ["Multiline", ["loss/train", "loss/val"]],
    },
}

class IRISCCCDDPMLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.fill_value = hparams['fill_value']    
        self.learning_rate = hparams['learning_rate']
        self.runs_dir = hparams['runs_dir']
        self.n_steps = hparams['n_steps']
        self.min_beta = hparams['min_beta']
        self.max_beta = hparams['max_beta']
        self.scheduler_step_size = hparams['scheduler_step_size']
        self.scheduler_gamma = hparams['scheduler_gamma']
        self.output_norm = hparams['output_norm']
        os.makedirs(self.runs_dir, exist_ok=True)

        self.loss = nn.MSELoss()  
        #self.loss = MaskedMSELoss(ignore_value = hparams['fill_value'])
        self.metrics_dict = nn.ModuleDict({
                    "rmse": MaskedRMSE(ignore_value = self.fill_value),  
                    "mae": MaskedMAE(ignore_value = self.fill_value)
                })

        self.model = CDDPM(n_steps=self.n_steps, 
                           min_beta=self.min_beta, 
                           max_beta=self.max_beta, 
                           encode_conditioning_image=False, 
                           in_ch=hparams['in_channels'])
        
        self.denorm = DeMinMaxNormalisation(hparams['sample_dir'], self.output_norm)

        self.test_metrics = {}
        self.train_step_outputs = []
        self.val_step_outputs = []

        self.save_hyperparameters()
        self.epoch_start_time = None

    def configure_model(self) -> None:
        self.model.betas = self.model.betas.to(self.device)
        self.model.alpha_bars = self.model.alpha_bars.to(self.device)
        self.model.alphas = self.model.alphas.to(self.device)

    def forward(self, x, y):
        # x = conditionning image
        # y = image to noise
        b = x.size(0)
        eta = torch.randn_like(y)
        t = torch.randint(1, self.model.n_steps, (b,), device=self.device)
        noisy_images = self.model(y, t, eta)
        eta_theta = self.model.backward(noisy_images, t.reshape(b, -1), x)
        return eta, eta_theta


    def on_train_start(self):
        self.logger.experiment.add_custom_scalars(layout)
        self.logger.log_hyperparams(vars(self.hparams))

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def common_step(self, x, y):
        eta, eta_theta = self(x, y)
        loss = torch.sqrt(self.loss(eta_theta, eta))
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.common_step(x, y)
        print('loss =', loss)
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
        loss = self.common_step(x, y)
        self.val_step_outputs.append(loss)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.val_step_outputs).mean()
        self.logger.experiment.add_scalar("loss/val", epoch_average, self.current_epoch)
        self.val_step_outputs.clear()


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.sampling(start_t=200, 
                                        conditioning_image=x.to(self.device), 
                                        eta = None)
        
        if self.output_norm is True:
            x[0,...], y[0,...] = self.denorm((x[0,...], y[0,...]))
            y_hat[0,...] = self.denorm((False, y_hat[0,...]))

        batch_dict = {}
        for metric_name, metric in self.metrics_dict.items():
            metric.update(y_hat, y)
            batch_dict[metric_name] = metric.compute()
            self.logger.experiment.add_scalar(metric_name, metric.compute(), batch_idx)
            metric.reset()
        self.test_metrics[batch_idx] = batch_dict

        if batch_idx == 0: 
            y[y == self.fill_value] = torch.nan
            x[x == self.fill_value] = torch.nan
            y_hat_mask = y_hat
            y_hat_mask[torch.isnan(y)] = torch.nan 
            fig, ax = plt.subplots()
            vmin, vmax = np.nanmin(y.cpu().numpy()), np.nanmax(y.cpu().numpy())
            levels = np.round(np.linspace(vmin, vmax, 11)).astype(int)
            cs = ax.contourf(y[0,0,:,:].cpu().detach().numpy(), cmap='OrRd', levels=levels)
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_y_0', fig)
    
            fig, ax = plt.subplots()
            cs = ax.contourf(x[0,-1,:,:].cpu().detach().numpy(), cmap='OrRd', levels=levels)
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_x_0', fig)

            fig, ax = plt.subplots()
            cs = ax.contourf(y_hat[0,0,:,:].cpu().detach().numpy(), cmap='OrRd')
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_yhat_raw_0', fig)

            fig, ax = plt.subplots()
            cs = ax.contourf(y_hat_mask[0,0,:,:].cpu().detach().numpy(), cmap='OrRd', levels=levels)
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]


