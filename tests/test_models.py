import torch
import pytest
from iriscc.models.unet import UNet
from iriscc.models.miniunet import MiniUNet
from iriscc.models.denoising_unet import DenoisingUNet
from iriscc.models.swin2sr import Swin2SR
from iriscc.models.miniswinunetr import MiniSwinUNETR

def test_unet_smoke():
    model = UNet(in_channels=2, out_channels=1)
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)

def test_mini_unet_smoke():
    model = MiniUNet(in_channels=2, out_channels=1)
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)

def test_denoising_unet_smoke():
    # DenoisingUNet expects (x, t) where t is timestep
    model = DenoisingUNet(in_channels=2, out_channels=1)
    x = torch.randn(1, 2, 64, 64)
    t = torch.randint(0, 1000, (1,))
    y = model(x, t)
    assert y.shape == (1, 1, 64, 64)

def test_swin2sr_smoke():
    # Swin2SR might have specific input requirements depending on implementation
    # Assuming standard forward(x)
    model = Swin2SR(in_channels=2, out_channels=1)
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)

def test_miniswinunetr_smoke():
    model = MiniSwinUNETR(in_channels=2, out_channels=1, img_size=(64, 64))
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)
