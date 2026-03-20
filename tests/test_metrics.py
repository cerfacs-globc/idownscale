import torch
import pytest
from iriscc.metrics import RMSE, Bias, CorrelationSpatial, CorrelationTemporal

def test_rmse():
    rmse = RMSE()
    x = torch.tensor([[[2.0, 4.0]]])
    y = torch.tensor([[[1.0, 2.0]]])
    # diff [1, 2], squared [1, 4], mean 2.5, sqrt(2.5) approx 1.58
    val = rmse(x, y)
    assert torch.isclose(val, torch.tensor(2.5**0.5))

def test_bias():
    bias = Bias()
    x = torch.tensor([[[2.0, 4.0]]])
    y = torch.tensor([[[1.0, 2.0]]])
    # diff [1, 2], mean 1.5
    val = bias(x, y)
    assert val == 1.5

def test_correlation_spatial():
    corr = CorrelationSpatial()
    x = torch.tensor([[[1.0, 2.0, 3.0]]])
    y = torch.tensor([[[1.0, 2.0, 3.0]]])
    # correlation of [1,2,3] with [1,2,3] is 1.0
    val = corr(x, y)
    assert torch.isclose(val, torch.tensor(1.0))

def test_correlation_temporal():
    corr = CorrelationTemporal()
    # CorrelationTemporal expects a list of samples or a batch with time dimension?
    # Let's check metrics.py for implementation
    # Assuming (batch, channels, height, width) or similar
    x = torch.tensor([[[1.0]], [[2.0]], [[3.0]]]) # time=3, chan=1, h=1, w=1
    y = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    val = corr(x, y)
    assert torch.isclose(val, torch.tensor(1.0))
