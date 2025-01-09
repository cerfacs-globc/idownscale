import sys
sys.path.append('.')

import torch.nn as nn

class MaskedMSELoss(nn.Module):
    def __init__(self, ignore_value):
        super(MaskedMSELoss, self).__init__()
        self.ignore_value = ignore_value

    def forward(self, y_hat, y):
        mask = (y != self.ignore_value).float()
        squared_error = (y_hat - y) ** 2
        masked_error = squared_error * mask
        loss = masked_error.sum() / (mask.sum())
        return loss