import sys
sys.path.append('.')

import torch
import torch.nn as nn

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
        loss = masked_error.sum() / (mask.sum())
        return loss