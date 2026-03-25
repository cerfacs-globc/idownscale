"""
Hyperparameters for the IRISCC project.

date : 16/07/2025
author : Zoé GARCIA
"""

import sys
sys.path.append('.')

from iriscc.settings import RUNS_DIR, CONFIG


class IRISCCHyperParameters():
    def __init__(self, exp: str = 'exp5'):
        self.exp_name = exp
        conf = CONFIG[exp]
        
        self.img_size = conf.get('shape', (64, 64))
        self.in_channels = len(conf.get('input_vars', []))
        self.mask = 'target'
        self.learning_rate = conf.get('learning_rate', 0.0008)
        self.batch_size = conf.get('batch_size', 32)
        self.max_epoch = conf.get('max_epoch', 30)
        self.model = conf.get('model', 'unet')
        self.exp = f"{exp}/{self.model}_all"
        self.runs_dir = RUNS_DIR / self.exp
        self.sample_dir = conf['dataset']
        self.fill_value = -1.
        self.domain = 'france'
        self.domain_crop = None
        self.channels = conf['channels']
        self.loss = conf.get('loss', 'masked_gamma_mae') # masked_mse or masked_gamma_mae
        self.dropout = conf.get('dropout', 0.0)
        
        # Diffusion hparams
        self.n_steps = 200
        self.min_beta = 1e-4
        self.max_beta = 0.02
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.1
        self.output_norm = False
