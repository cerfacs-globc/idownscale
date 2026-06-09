import torch
import pytest
from iriscc.models.unet import UNet
from iriscc.models.miniunet import MiniUNet
from iriscc.models.denoising_unet import CUNet
from iriscc.models.cddpm import CDDPM
from iriscc.lightning_module_ddpm import IRISCCCDDPMLightningModule
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

def test_cddpm_smoke():
    model = CDDPM(n_steps=8, in_ch=3)
    model.set_device("cpu")
    conditioning = torch.randn(1, 2, 32, 32)
    target = torch.randn(1, 1, 32, 32)
    t = torch.randint(1, model.n_steps, (1,), device=torch.device("cpu"))
    noisy = model(target, t)
    assert noisy.shape == (1, 1, 32, 32)
    eta_theta = model.backward(noisy, t.reshape(1, -1), conditioning)
    assert eta_theta.shape == (1, 1, 32, 32)
    sample = model.sampling(start_t=model.n_steps, conditioning_image=conditioning)
    assert sample.shape == (1, 1, 32, 32)

def test_cddpm_denoiser_output_is_unbounded_regression_head():
    model = CDDPM(n_steps=8, in_ch=3)
    last = model.network.conv
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(-2.0)
    conditioning = torch.randn(1, 2, 32, 32)
    noisy = torch.randn(1, 1, 32, 32)
    t = torch.ones(1, 1, dtype=torch.long)
    eta_theta = model.backward(noisy, t, conditioning)
    assert torch.all(eta_theta < 0.0)

def test_cddpm_lightning_module_forward_smoke(tmp_path):
    hparams = {
        "fill_value": 0.0,
        "learning_rate": 8e-4,
        "runs_dir": tmp_path,
        "n_steps": 8,
        "min_beta": 1e-4,
        "max_beta": 0.02,
        "scheduler_step_size": 50,
        "scheduler_gamma": 0.1,
        "output_norm": False,
        "in_channels": 2,
        "sample_dir": tmp_path,
    }
    module = IRISCCCDDPMLightningModule(hparams)
    x = torch.randn(2, 2, 32, 32)
    y = torch.randn(2, 1, 32, 32)
    eta, eta_theta = module(x, y)
    assert eta.shape == (2, 1, 32, 32)
    assert eta_theta.shape == (2, 1, 32, 32)

def test_swin2sr_smoke():
    pytest.skip("External Swin2SR model test skipped")

def test_miniswinunetr_smoke():
    pytest.skip("MiniSwinUNETR implementation is incompatible with installed MONAI version")
    model = MiniSwinUNETR(in_channels=2, out_channels=1, img_size=(64, 64))
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)
