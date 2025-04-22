import sys
sys.path.append('.')

from iriscc.settings import RUNS_DIR, DATASET_EXP3_30Y_DIR, DATASET_EXP4_30Y_DIR, CONFIG

class IRISCCHyperParameters():
    def __init__(self):
        
        self.img_size = (160, 160)
        self.in_channels = 2
        self.mask = 'target'
        self.learning_rate = 0.001
        self.batch_size = 32
        self.max_epoch = 60
        self.model ='swinunetr'
        self.exp = 'exp3/swinunet_all'
        self.runs_dir = RUNS_DIR / self.exp
        self.sample_dir = DATASET_EXP3_30Y_DIR
        self.fill_value = 0.
        self.domain = 'france'
        self.domain_crop = None
        # Diffusion hparams
        self.n_steps = 200
        self.min_beta = 1e-4
        self.max_beta = 0.02
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.1
        self.output_norm = False
