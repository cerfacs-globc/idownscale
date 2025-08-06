"""
Loss functions for training models.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Gamma
from pathlib import Path
from iriscc.plotutils import plot_test
from iriscc.settings import GRAPHS_DIR

class MaskedMSELoss(nn.Module):
    """
    MaskedMSELoss is a custom loss function that computes the Mean Squared Error (MSE)
    while ignoring specific target values.

    Attributes:
        ignore_value (float): The value in the target tensor `y` to be ignored during
            loss computation.
    """
    def __init__(self, ignore_value):
        super(MaskedMSELoss, self).__init__()
        self.ignore_value = ignore_value

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the masked mean squared error loss between the predicted values (y_hat)
        and the target values (y), ignoring a specific value.

        Args:
            y_hat (torch.Tensor): Predicted values with shape (N, ...), where N is the batch size.
            y (torch.Tensor): Target values with shape (N, ...), where N is the batch size.

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
        mask = (y != self.ignore_value).float()
        squared_error = (y_hat - y) ** 2
        masked_error = squared_error * mask
        loss = masked_error.sum() / (mask.sum()) # Mean over non-masked values
        return torch.sqrt(loss)
    


class MaskedGammaMAELoss(nn.Module):
    """
    MaskedGammaMAELoss is a PyTorch loss function that computes the masked mean absolute error
    between predicted values and target values, incorporating a Gamma distribution for modeling.
    Attributes:
        ignore_value (float): The value to ignore in the loss computation.
        alpha (torch.Tensor): Concentration parameter for the Gamma distribution.
        beta (torch.Tensor): Rate parameter for the Gamma distribution.
    Methods:
        forward(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            Computes the loss between the predicted values (y_hat) and the target values (y).
            Applies a mask to ignore specified values during loss calculation.
    Following Antoine Doury's code : https://github.com/antoinedoury/RCM-Emulator
    """
    def __init__(self, ignore_value:float, sample_dir:Path):
        super(MaskedGammaMAELoss, self).__init__()
        gamma_params_file = dict(np.load(sample_dir/ 'gamma_params.npz', allow_pickle=True))
        alpha, beta = gamma_params_file['alpha'], gamma_params_file['beta']
        alpha, beta = torch.tensor(alpha, dtype=torch.float32), torch.tensor(beta, dtype=torch.float32)
        alpha[torch.isnan(alpha)] = 1.0  # Replace NaN with 1.0 or random value > 0 to match Gamma distribution requirements
        alpha = torch.unsqueeze(alpha, dim=0)  # Ensure alpha is a tensor of shape (1, h, w)
        self.register_buffer("alpha", alpha) 

        beta[torch.isnan(beta)] = 1.0    # Replace NaN with 1.0
        beta = torch.unsqueeze(beta, dim=0)
        self.register_buffer("beta", beta)

        self.ignore_value = ignore_value
        
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gamma_dist = Gamma(concentration=self.alpha.to(y.device), rate=self.beta.to(y.device))
        mask = (y != self.ignore_value).float()
        y_copy = y.clone()
        y_copy[y_copy == self.ignore_value] = 0.0  # Replace ignore_value with 0 for Gamma dist computation
        cdf_values = gamma_dist.cdf(y_copy)
        loss_2D = torch.abs(y_copy - y_hat) + cdf_values**2 * torch.max(torch.zeros_like(y_copy), y_copy - y_hat)
        masked_loss_2D = loss_2D * mask  # Apply mask to the loss
        loss = masked_loss_2D.sum() / mask.sum() # Mean over non-masked values
        return loss

if __name__ == '__main__':
    alpha = torch.nan * torch.ones(64,64)
    beta = torch.nan * torch.ones(64,64)
    loss_fn = MaskedGammaMAELoss(ignore_value=-1.0, alpha=alpha, beta=beta)

    y_hat = torch.rand(1,1,64,64) * 10.0  # Predicted values
    y = torch.rand(1,1,64,64) * 10.0  # Target values

    loss = loss_fn(y_hat, y)
    print("Loss:", loss.item())