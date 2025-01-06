import sys
sys.path.append('.')

from iriscc.settings import RUNS_DIR, DATASET_EXP1_DIR

class IRISCCHyperParameters():
    def __init__(self):
        
        self.in_channels = 3
        self.learning_rate = 0.003
        self.batch_size = 32
        self.max_epoch = 2
        self.model = 'unet'
        self.exp = 'exp0/mask/none'
        self.runs_dir = RUNS_DIR / self.exp
        self.sample_dir = DATASET_EXP1_DIR
        self.mask = 'none'
        self.fill_value = 0.
        self.landseamask = False # Add binary mask to inputs and channels

