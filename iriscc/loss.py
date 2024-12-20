import sys
sys.path.append('.')

import torch
import torch.nn as nn

class WeightedRMSELoss(nn.Module):
    ''' Found on pytorch, noramlized y ???'''
    def init(self,weights=None):
        super(WeightedRMSELoss, self).init()
        self.weights=weights #Initialize weights

    def forward(self, y_pred, y_true, weights=None):
        if weights is None: #We build the class this way so you can call the weights when creating         the loss or when applying it.
            if self.weights is None: #No input weights when first creating the class.
                weights = torch.ones_like(y_true)*1/y_true.size(1)  # Default to ones if no weights are                                provided
            else: #If a weights vector is provided.
                weights = self.weights/torch.sum(self.weights) #Make sure our weights vector is                                 normalized so the sum of all elements is equal to 1.
        # Calculate squared errors
        squared_errors = (y_pred - y_true) ** 2

        # Apply weights to squared errors
        weighted_squared_errors = squared_errors * weights #By making sure our weights vector         is normalized this always works, if no weights are provided this is identical to using RMSE.

        # Take square root to get RMSE 
        rmse_single = torch.sqrt(torch.sum(weighted_squared_errors,dim=1)) #dim=1 makes sure         we're calculating one rmse value for each row.
        rmse=torch.mean(rmse_single) #Now we're reducing over the batch dimension.

        return rmse 
    
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