"""
@Time    : 2020/9/29 12:22
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : predictor.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn


class StatePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StatePredictor, self).__init__()

        self.pred = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.pred(x)
