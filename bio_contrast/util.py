"""
@Time    : 2020/11/9 16:54
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : util.py
@Software: PyCharm
@Desc    : 
"""

import warnings

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def setup_seed(seed):
    warnings.warn(f'You have chosen to seed ({seed}) training. This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_performance(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')

    return {'accuracy': accuracy, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
