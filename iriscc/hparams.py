"""
Hyperparameters for the IRISCC project.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys

sys.path.append(".")

from pathlib import Path

from iriscc.settings import RUNS_DIR, CONFIG


class IRISCCHyperParameters:
    def __init__(
        self,
        *,
        exp: str = "exp5",
        run_name: str | None = None,
        model: str = "unet",
        learning_rate: float = 0.0008,
        batch_size: int = 32,
        max_epoch: int = 30,
        loss: str | None = None,
        dropout: float = 0.0,
        output_norm: bool = False,
        mask: str = "target",
        fill_value: float = 0.0,
        sample_dir: str | Path | None = None,
        statistics_dir: str | Path | None = None,
        seed: int | None = None,
        n_steps: int = 200,
        output_range: str = "zero_one",
    ):
        cfg = CONFIG[exp]
        self.img_size = cfg["shape"]
        self.in_channels = len(cfg["channels"]) - 1
        self.mask = mask
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.model = model
        self.run_name = run_name or "unet_all"
        self.exp_name = exp
        self.exp = f"{exp}/{self.run_name}"
        self.runs_dir = RUNS_DIR / self.exp
        self.sample_dir = Path(sample_dir) if sample_dir is not None else cfg["dataset"]
        self.statistics_dir = Path(statistics_dir) if statistics_dir is not None else self.sample_dir
        self.seed = seed
        self.fill_value = fill_value
        self.domain = "france"
        self.domain_crop = None
        self.channels = cfg["channels"]
        if loss is None:
            self.loss = "masked_gamma_mae" if "pr" in cfg["target_vars"] else "masked_mse"
        else:
            self.loss = loss
        self.dropout = dropout

        # Diffusion hparams
        self.n_steps = n_steps
        self.min_beta = 1e-4
        self.max_beta = 0.02
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.1
        self.output_norm = output_norm
        self.output_range = output_range
