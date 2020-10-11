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
    def __init__(self, input_channels, hidden_channels, feature_dim, pred_steps, num_seq, batch_size, kernel_sizes):
        super(SleepContrast, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.batch_size = batch_size
        self.kernel_sizes = kernel_sizes
        self.num_seq = num_seq

        self.targets = None

        # Local Encoder
        self.encoder = ResNet(input_channels, hidden_channels, feature_dim, kernel_sizes=kernel_sizes)

        # Memory bank
        #         memory_bank = torch.randn(total_size, output_length)
        #         self.register_buffer('memory_bank', memory_bank)

        # Aggregator
        self.gru = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2)

        # Predictor
        self.predictor = StatePredictor(input_dim=feature_dim, output_dim=feature_dim)

    #     def _initialize_weights(self, module):
    #         for name, param in module.named_parameters():
    #             if 'bias' in name:
    #                 nn.init.constant_(param, 0.0)
    #             elif 'weight' in name:
    #                 nn.init.orthogonal_(param, 0.1)

    def compute_targets(self, recompute=False):
        if recompute or self.targets is None:
            self.targets = torch.zeros(self.batch_size, self.pred_steps, self.num_seq, self.batch_size).long()
            for i in range(self.batch_size):
                for j in range(self.pred_steps):
                    self.targets[i, j, self.num_seq - self.pred_steps + j, i] = 1

            self.targets = self.targets.cuda()
            self.targets = self.targets.view(self.batch_size * self.pred_steps, self.num_seq * self.batch_size)
            self.targets = self.targets.argmax(dim=1)
            return self.targets
        else:
            return self.targets

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        (batch, num_seq, channel, seq_len) = x.shape
        x = x.view(batch * num_seq, channel, seq_len)
        feature = self.encoder(x)
        feature = feature.view(batch, num_seq, self.feature_dim)  # (batch, num_seq, feature_dim)

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
        feature = feature.transpose(0, 2).contiguous()  # (feature_dim, num_seq, batch)
        pred = pred.contiguous()

        score = torch.einsum('ijk,kmn->ijmn', [pred, feature])  # (batch, pred_step, num_seq, batch)
        score = score.view(batch * self.pred_steps, num_seq * batch)

        return score
