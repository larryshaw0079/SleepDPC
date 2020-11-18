"""
@Time    : 2020/9/29 12:22
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : gru.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)

    def forward(self, x, h_0):
        # x:   (batch, seq_len,    input_size)
        # h_0: (num_layers, batch, hidden_size)

        out, h_n = self.gru(x, h_0)

        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        return out, h_n

    def init_hidden(self, batch_size, device):
        hidden_states = torch.randn(self.num_layers, batch_size, self.hidden_size)
        hidden_states = hidden_states.to(device)
        return hidden_states
