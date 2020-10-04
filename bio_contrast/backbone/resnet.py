"""
@Time    : 2020/9/29 12:22
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : resnet.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_sizes=[7, 11, 7], stride=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        assert len(kernel_sizes) == 3

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=kernel_sizes[0], stride=1,
                      padding=kernel_sizes[0] // 2, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

        # Only conv2 degrades the scale
        self.conv2 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_sizes[1], stride=stride,
                      padding=kernel_sizes[1] // 2, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_sizes[2], stride=1,
                      padding=kernel_sizes[2] // 2, bias=False),
            nn.BatchNorm1d(output_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        # If stride == 1, the length of the time dimension will not be changed
        # If input_channels == output_channels, the number of channels will not be changed
        # If the channels are mismatch, the conv1d is used to upgrade the channel
        # If the time dimensions are mismatch, the conv1d is used to downsample the scale
        self.downsample = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(output_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Downsampe is an empty list if the size of inputs and outputs are same
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes, kernel_sizes=[7, 11, 7]):
        super(ResNet, self).__init__()

        # The first convolution layer
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(input_channels, hidden_channels, kernel_size=15, stride=2, padding=7, bias=False),
#             nn.BatchNorm1d(hidden_channels),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         )
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Residual layers
        self.layer1 = self.__make_layer(ResidualBlock, hidden_channels, hidden_channels, 2, kernel_sizes, stride=1)
        self.layer2 = self.__make_layer(ResidualBlock, hidden_channels, hidden_channels*2, 2, kernel_sizes, stride=2)
        self.layer3 = self.__make_layer(ResidualBlock, hidden_channels*2, hidden_channels*4, 2, kernel_sizes, stride=2)
        self.layer4 = self.__make_layer(ResidualBlock, hidden_channels*4, hidden_channels*8, 2, kernel_sizes, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1) # Pooling operation computes the average of the last dimension (time dimension)

        # A dense layer for output
        self.fc = nn.Linear(hidden_channels*8, num_classes)

        # Initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

    def __make_layer(self, block, input_channels, output_channels, num_blocks, kernel_sizes, stride):
        layers = []
        layers.append(block(input_channels, output_channels, kernel_sizes, stride=stride))
        for i in range(1, num_blocks):
            layers.append(block(output_channels, output_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        L_out = floor[(L_in + 2*padding - kernel) / stride + 1]
        """
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)

        return out
