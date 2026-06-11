"""
Shared helpers for loading trained Lightning modules and running inference.
"""

from __future__ import annotations

from pathlib import Path

import torch

from iriscc.lightning_module import IRISCCLightningModule
from iriscc.lightning_module_ddpm import IRISCCCDDPMLightningModule


def load_trained_module(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    hparams = hyper_parameters.get("hparams", hyper_parameters)
    model_name = hparams.get("model", "unet")
    module_cls = IRISCCCDDPMLightningModule if model_name == "cddpm" else IRISCCLightningModule

    module = module_cls.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    module.to(device)
    module.eval()

    if model_name == "cddpm":
        module.model.set_device(str(device))

    return module, hparams


def predict_tensor(module, x: torch.Tensor, hparams: dict, device: str | torch.device, num_samples: int = 1):
    model_name = hparams.get("model", "unet")
    with torch.no_grad():
        if model_name == "cddpm":
            draws = [
                module.model.sampling(
                    start_t=module.model.n_steps,
                    conditioning_image=x.to(device),
                    eta=None,
                )
                for _ in range(max(1, int(num_samples)))
            ]
            if len(draws) == 1:
                return draws[0]
            return torch.stack(draws, dim=0).mean(dim=0)
        return module(x.to(device))
