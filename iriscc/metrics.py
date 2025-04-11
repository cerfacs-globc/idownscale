import torch
from torch import nn
from torchmetrics import Metric

class MaskedRMSE(Metric):
    def __init__(self, ignore_value=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_value = ignore_value
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None):
        if self.ignore_value is not None:
            mask = (target != self.ignore_value).float()
        else:
            mask = torch.ones_like(target)

        if weight is None:
            weight = torch.ones_like(target)

        effective_weight = mask * weight
        squared_error = (preds - target) ** 2
        self.sum_squared_error += (squared_error * effective_weight).sum()
        self.total_weight += effective_weight.sum()

    def compute(self):
        mean_squared_error = self.sum_squared_error / (self.total_weight + 1e-8)
        return torch.sqrt(mean_squared_error)


class MaskedMAE(Metric):
    def __init__(self, ignore_value=None, **kwargs):
        super().__init__(**kwargs)
        self.ignore_value = ignore_value
        self.add_state("sum_absolute_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_weight", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None):
        if self.ignore_value is not None:
            mask = (target != self.ignore_value).float()
        else:
            mask = torch.ones_like(target)

        if weight is None:
            weight = torch.ones_like(target)

        effective_weight = mask * weight
        absolute_error = torch.abs(preds - target)
        self.sum_absolute_error += (absolute_error * effective_weight).sum()
        self.total_weight += effective_weight.sum()

    def compute(self):
        mean_absolute_error = self.sum_absolute_error / (self.total_weight + 1e-8)
        return mean_absolute_error