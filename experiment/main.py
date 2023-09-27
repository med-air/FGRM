import torch
import pandas as pd

import configs 
from methods.base_method import BaseMethod

import torch.backends.cudnn as cudnn
import numpy as np
import random
import os

if __name__ == '__main__':
    seed = 0
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    method = BaseMethod(configs)
    method.run_FGRM()
