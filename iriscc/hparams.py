"""
Hyperparameters for the IRISCC project.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

from iriscc.settings import RUNS_DIR, CONFIG


class IRISCCHyperParameters():
    def __init__(
        self,
        *,
        exp: str = 'exp5',
        run_name: str | None = None,
        model: str = 'unet',
        learning_rate: float = 0.0008,
        batch_size: int = 32,
        max_epoch: int = 30,
        loss: str | None = None,
        dropout: float = 0.0,
        output_norm: bool = False,
        mask: str = 'target',
        fill_value: float = 0.0,
    ):
        cfg = CONFIG[exp]
        self.img_size = cfg['shape']
        self.in_channels = len(cfg['input_vars'])
        self.mask = mask
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.model = model
        self.run_name = run_name or 'unet_all'
        self.exp_name = exp
        self.exp = f'{exp}/{self.run_name}'
        self.runs_dir = RUNS_DIR / self.exp
        self.sample_dir = cfg['dataset']
        self.fill_value = fill_value
        self.domain = 'france'
        self.domain_crop = None
        self.channels = cfg['channels']
        if loss is None:
            self.loss = 'masked_gamma_mae' if 'pr' in cfg['target_vars'] else 'masked_mse'
        else:
            self.loss = loss
        self.dropout = dropout

        # Diffusion hparams
        self.n_steps = 200
        self.min_beta = 1e-4
        self.max_beta = 0.02
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.1
        self.output_norm = output_norm
