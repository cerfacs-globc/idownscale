import sys
sys.path.append('.')

import glob
import numpy as np
from iriscc.settings import DATASET_EXP1_DIR

list_sample = np.sort(glob.glob(str(DATASET_EXP1_DIR/'sample*.npz')))
nb = len(list_sample)
print(len(list_sample))
train_end = int(0.6 * nb) 
val_end = train_end + int(0.2 * nb)
train = list_sample[:train_end]
val = list_sample[train_end:val_end]
test = list_sample[val_end:]
