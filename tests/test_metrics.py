import torch
import pytest
from iriscc.metrics import MaskedRMSE, MaskedMAE

def test_masked_rmse():
    rmse = MaskedRMSE()
    x = torch.tensor([[[2.0, 4.0]]])
    y = torch.tensor([[[1.0, 2.0]]])
    # diff [1, 2], squared [1, 4], mean 2.5, sqrt(2.5) approx 1.58
    rmse.update(x, y)
    val = rmse.compute()
    assert torch.isclose(val, torch.tensor(2.5**0.5))

def test_masked_mae():
    mae = MaskedMAE()
    x = torch.tensor([[[2.0, 4.0]]])
    y = torch.tensor([[[1.0, 2.0]]])
    # diff [1, 2], absolute [1, 2], mean 1.5
    mae.update(x, y)
    val = mae.compute()
    assert torch.isclose(val, torch.tensor(1.5))
