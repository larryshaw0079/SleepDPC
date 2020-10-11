"""
@Time    : 2020/9/29 12:25
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : sleep_classifier.py
@Software: PyCharm
@Desc    : 
"""
import torch.nn as nn

from ..backbone import ResNet, GRU


class SleepClassifier(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes, feature_dim, pred_steps, num_seq, batch_size,
                 kernel_sizes):
        super(SleepClassifier, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.num_seq = num_seq
        self.batch_size = batch_size
        self.kernel_sizes = kernel_sizes

        # Local Encoder
        self.encoder = ResNet(input_channels, hidden_channels, feature_dim, kernel_sizes=kernel_sizes)

        # Aggregator
        self.gru = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2)

        # Classifier
        self.mlp = nn.Sequential(
            nn.ReLU(inplace=True),
            #             nn.Linear(feature_dim, feature_dim),
            #             nn.BatchNorm1d(feature_dim),
            #             nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_classes)
        )

    def freeze_parameters(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.gru.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        (batch, num_seq, channel, seq_len) = x.shape
        x = x.view(batch * num_seq, channel, seq_len)
        feature = self.encoder(x)
        feature = feature.view(batch, num_seq, self.feature_dim)  # (batch, num_seq, feature_dim)

        # Get context feature
        h_0 = self.gru.init_hidden(self.batch_size)
        # context: (batch, num_seq, hidden_size)
        # h_n:     (num_layers, batch, hidden_size)
        context, h_n = self.gru(feature[:, :-self.pred_steps, :], h_0)

        context = context[:, -1, :]
        #         out = self.relu(context)
        out = self.mlp(context)

        return out
