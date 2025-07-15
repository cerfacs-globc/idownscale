import sys
sys.path.append('.')

import numpy as np
import torch

from iriscc.settings import RUNS_DIR, CONFIG
from iriscc.loss import MaskedMSELoss, MaskedGammaMAELoss

def get_gamma_params(sample_dir):
    data = dict(np.load(sample_dir/'gamma_params.npz', allow_pickle=True))
    alpha = data['alpha']
    beta = data['beta']
    return torch.Tensor(alpha), torch.Tensor(beta)


class IRISCCHyperParameters():
    def __init__(self):
        exp = 'exp6'
        self.img_size = CONFIG[exp]['shape']
        self.in_channels = 2
        self.mask = 'target'
        self.learning_rate = 0.001
        self.batch_size = 32
        self.max_epoch = 30
        self.model ='unet'
        self.exp = f'{exp}/unet_all'
        self.runs_dir = RUNS_DIR / self.exp
        self.sample_dir = CONFIG[exp]['dataset']
        self.fill_value = -1.
        #alpha, beta = get_gamma_params(self.sample_dir)
        #self.loss = MaskedGammaMAELoss(ignore_value = self.fill_value,
        #                               alpha = alpha,
        #                               beta = beta) ### modifier loss dans hparams.yaml
        self.domain = 'france'
        self.domain_crop = None
        # Diffusion hparams
        self.n_steps = 200
        self.min_beta = 1e-4
        self.max_beta = 0.02
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.1
        self.output_norm = False
