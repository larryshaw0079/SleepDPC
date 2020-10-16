"""
@Time    : 2020/9/29 12:23
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : sleep_contrast.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn

from ..backbone import ResNet, GRU, StatePredictor


class SleepContrast(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, pred_steps, num_seq, batch_size, relative_position,
                 kernel_sizes):
        super(SleepContrast, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.batch_size = batch_size
        self.kernel_sizes = kernel_sizes
        self.num_seq = num_seq
        self.relative_position = relative_position

        # Local Encoder
        self.encoder = ResNet(input_channels, hidden_channels, feature_dim, kernel_sizes=kernel_sizes)

        # Aggregator
        self.gru = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2)

        # Predictor
        self.predictor = StatePredictor(input_dim=feature_dim, output_dim=feature_dim)

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        (batch, num_seq, channel, seq_len) = x.shape
        x = x.view(batch * num_seq, channel, seq_len)
        feature = self.encoder(x)
        feature = feature.view(batch, num_seq, self.feature_dim)  # (batch, num_seq, feature_dim)
        feature_trans = feature.transpose(0, 2).contiguous()

        if self.relative_position:
            position_score = torch.einsum('ijk,kmn->ijmn', [feature, feature_trans])
            position_score = position_score.view(batch * num_seq, num_seq * batch)

        # Get context feature
        h_0 = self.gru.init_hidden(self.batch_size)
        # out: (batch, num_seq, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, h_n = self.gru(feature[:, :-self.pred_steps, :], h_0)

        # Get predictions
        pred = []
        h_next = h_n
        c_next = out[:, -1, :].squeeze(1)
        for i in range(self.pred_steps):
            z_pred = self.predictor(c_next)
            pred.append(z_pred)
            c_next, h_next = self.gru(z_pred.unsqueeze(1), h_next)
            c_next = c_next[:, -1, :].squeeze(1)
        pred = torch.stack(pred, 1)  # (batch, pred_step, feature_dim)

        # Compute scores
        pred = pred.contiguous()

        cpc_score = torch.einsum('ijk,kmn->ijmn', [pred, feature_trans])  # (batch, pred_step, num_seq, batch)
        cpc_score = cpc_score.view(batch * self.pred_steps, num_seq * batch)

        if self.relative_position:
            return cpc_score, position_score
        else:
            return cpc_score
