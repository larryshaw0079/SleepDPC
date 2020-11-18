"""
@Time    : 2020/11/10 21:27
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : dcc.py
@Software: PyCharm
@Desc    : 
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..backbone import ResNet, MLP


class DCC(object):
    def __init__(self, num_seq, input_channel, input_length, feature_dim, batch_size, device, args):
        self.num_seq = num_seq
        self.input_channel = input_channel
        self.input_length = input_length
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.device = device

        self.encoder = ResNet()
        self.classifier = MLP()

        self.encoder = self.encoder.to(device)
        self.classifier = self.classifier.to(device)

    def pretrain(self, data_loader, args):
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(self.encoder.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(self.encoder.parameters(), lr=args.learning_rate, betas=(0.9, 0.98),
                                   eps=1e-09, weight_decay=args.weight_decay, amsgrad=True)
        else:
            raise ValueError('Invalid optimizer option!')

        criterion = nn.CrossEntropyLoss()

        self.encoder.train()
        target = self.__compute_target()
        for epoch in range(args.pretrain_epochs):
            losses = []

            for x in data_loader:
                x = x.to(self.device)
                x = x.view(self.batch_size * self.num_seq, -1)

                z = self.encoder(x)
                z = z.view(self.batch_size, self.num_seq, -1)
                score = self.__compute_score(z)
                loss = criterion(score, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            if (epoch + 1) % args.disp_interval == 0:
                print(f'EPOCH: [{epoch + 1}/{args.epochs}] Loss: {np.mean(losses):.6f}')

    def finetune(self, data_loader, args):
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(self.classifier.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(self.classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.98),
                                   eps=1e-09, weight_decay=args.weight_decay, amsgrad=True)
        else:
            raise ValueError('Invalid optimizer option!')

        criterion = nn.CrossEntropyLoss()

        self.encoder.eval()
        self.classifier.train()
        self.__freeze_parameters()

        for epoch in range(args.finetune_epochs):
            losses = []

            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                x = x.view(self.batch_size * self.num_seq, -1)
                y = y.view(self.batch_size * self.num_seq)

                z = self.encoder(x)
                y_hat = self.classifier(z)
                loss = criterion(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            if (epoch + 1) % args.disp_interval == 0:
                print(f'EPOCH: [{epoch + 1}/{args.epochs}] Loss: {np.mean(losses):.6f}')

    def evaluate(self, data_loader, args):
        self.encoder.eval()
        self.classifier.eval()
        self.__freeze_parameters()

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                x = x.view(self.batch_size * self.num_seq, -1)
                y = y.view(self.batch_size * self.num_seq)

                z = self.encoder(x)
                y_hat = self.classifier(z)

    def save(self):
        pass

    def load(self):
        pass

    def __repr__(self):
        return self.encoder.__repr__() + '\n' + self.classifier.__repr__()

    def __compute_score(self, z):
        scores = torch.einsum('ijk,mnk->ijnm', [z, z])  # (batch, num_seq, num_seq, batch)
        scores = scores.view(self.batch_size * self.num_seq, self.num_seq * self.batch_size)

        return scores

    def __compute_target(self):
        targets = torch.zeros(self.batch_size, self.num_seq, self.num_seq, self.batch_size).long()
        for i, j, k in itertools.product(range(self.batch_size), range(self.num_seq), range(self.num_seq)):
            targets[i, j, k, i] = 1

        targets = targets.to(self.device)
        targets = targets.view(self.batch_size * self.num_seq, self.num_seq * self.batch_size)
        targets = targets.argmax(dim=1)

        return targets

    def __freeze_parameters(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
