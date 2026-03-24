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
    # CUNet expects (x, t, conditionning_image)
    model = CUNet(n_steps=1000, in_channels=2, out_channels=1)
    x = torch.randn(1, 1, 64, 64)
    c = torch.randn(1, 2, 64, 64)
    t = torch.randint(0, 1000, (1,))
    y = model(x, t, c)
    assert y.shape == (1, 1, 64, 64)

def test_swin2sr_smoke():
    model = Swin2SR(in_chans=2, out_chans=1, window_size=8)
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)

def test_miniswinunetr_smoke():
    pytest.skip("MiniSwinUNETR implementation is incompatible with installed MONAI version")
    model = MiniSwinUNETR(in_channels=2, out_channels=1, img_size=(64, 64))
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)
