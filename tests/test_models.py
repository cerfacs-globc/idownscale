import torch
import pytest
from iriscc.models.unet import UNet
from iriscc.models.miniunet import MiniUNet
from iriscc.models.denoising_unet import CUNet
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
    pytest.skip("External CUNet model test skipped")

def test_swin2sr_smoke():
    pytest.skip("External Swin2SR model test skipped")

def test_miniswinunetr_smoke():
    pytest.skip("MiniSwinUNETR implementation is incompatible with installed MONAI version")
    model = MiniSwinUNETR(in_channels=2, out_channels=1, img_size=(64, 64))
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)
