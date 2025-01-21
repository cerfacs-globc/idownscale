import sys
sys.path.append('.')

from iriscc.settings import RUNS_DIR, DATASET_EXP1_DIR, DATASET_EXP1_30Y_DIR, DATASET_EXP1_6MB_DIR

class IRISCCHyperParameters():
    def __init__(self):
        
        self.img_size = (160,160)
        self.in_channels = 7
        self.mask = 'france'
        if self.mask != 'none':
            self.in_channels +=1
        self.learning_rate = 0.00001
        self.batch_size = 32
        self.max_epoch = 60
        self.model = 'swin2sr'
        self.exp = 'exp1/swin_6_members'
        self.runs_dir = RUNS_DIR / self.exp
        self.sample_dir = DATASET_EXP1_6MB_DIR
        self.fill_value = 0.
        

