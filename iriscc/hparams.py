import sys
sys.path.append('.')

from iriscc.settings import RUNS_DIR, DATASET_EXP1_DIR, DATASET_EXP1_30Y_DIR

class IRISCCHyperParameters():
    def __init__(self):
        
        self.in_channels = 3
        self.mask = 'france'
        if self.mask != 'none':
            self.in_channels +=1
        self.learning_rate = 0.001
        self.batch_size = 64
        self.max_epoch = 150
        self.model = 'unet'
        self.exp = 'exp1/30y'
        self.runs_dir = RUNS_DIR / self.exp
        self.sample_dir = DATASET_EXP1_30Y_DIR
        self.fill_value = 0.
        

