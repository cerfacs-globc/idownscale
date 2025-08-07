"""
Hyperparameters for the IRISCC project.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

from iriscc.settings import RUNS_DIR, CONFIG


class IRISCCHyperParameters():
    def __init__(self):
        exp = 'exp8'
        self.img_size = CONFIG[exp]['shape']
        self.in_channels = 13
        self.mask = 'target'
        self.learning_rate = 0.0008
        self.batch_size = 32
        self.max_epoch = 30
        self.model ='unet'
        self.exp = f'{exp}/unet'
        self.runs_dir = RUNS_DIR / self.exp
        self.sample_dir = CONFIG[exp]['dataset']
        self.fill_value = -1.
        self.domain = 'france'
        self.domain_crop = None
        self.channels = CONFIG[exp]['channels']
        self.loss = 'masked_gamma_mae' # masked_mse or masked_gamma_mae
        self.dropout = 0.0
        
        # Diffusion hparams
        self.n_steps = 200
        self.min_beta = 1e-4
        self.max_beta = 0.02
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.1
        self.output_norm = False
