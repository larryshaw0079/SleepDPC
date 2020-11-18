"""
@Time    : 2020/11/10 21:27
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : dpc.py
@Software: PyCharm
@Desc    : 
"""

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..backbone import ResNet, GRU, MLP
from ..util import get_performance


class DPC(object):
    def __init__(self, num_seq, input_channel, input_length, feature_dim, pred_steps, batch_size, device, args):
        self.num_seq = num_seq
        self.input_channel = input_channel
        self.input_length = input_length
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.batch_size = batch_size
        self.device = device

        self.encoder = ResNet()
        self.gru = GRU()
        self.predictor = MLP()
        self.classifier = MLP()

        self.encoder = self.encoder.to(device)
        self.gru = self.gru.to(device)
        self.predictor = self.predictor.to(device)
        self.classifier = self.classifier.to(device)

    def pretrain(self, data_loader, args):
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(itertools.chain(self.encoder.parameters(), self.gru.parameters(),
                                                  self.predictor.parameters()),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(itertools.chain(self.encoder.parameters(), self.gru.parameters(),
                                                   self.predictor.parameters()),
                                   lr=args.learning_rate, betas=(0.9, 0.98),
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

                (batch, num_seq, channel, seq_len) = x.shape
                x = x.view(batch * num_seq, channel, seq_len)

                z = self.encoder(x)
                z = z.view(batch, num_seq, self.feature_dim)  # (batch, num_seq, feature_dim)
                z = z.transpose(0, 2).contiguous()

                # Get context feature
                h_0 = self.gru.init_hidden(batch, self.device)
                # out: (batch, num_seq, hidden_size)
                # h_n: (num_layers, batch, hidden_size)
                out, h_n = self.gru(z[:, :-self.pred_steps, :], h_0)

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
                pred = pred.contiguous()

                # Compute scores
                score = torch.einsum('ijk,kmn->ijmn', [pred, z])  # (batch, pred_step, num_seq, batch)
                score = score.view(batch * self.pred_steps, num_seq * batch)

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
        self.gru.eval()
        self.predictor.eval()
        self.classifier.train()
        self.__freeze_parameters()

        for epoch in range(args.finetune_epochs):
            losses = []

            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)

                # x: (batch, num_seq, channel, seq_len)
                (batch, num_seq, channel, seq_len) = x.shape
                x = x.view(batch * num_seq, channel, seq_len)
                z = self.encoder(x)
                z = z.view(batch, num_seq, self.feature_dim)  # (batch, num_seq, feature_dim)

                # Get context feature
                h_0 = self.gru.init_hidden(batch, self.device)
                # context: (batch, num_seq, hidden_size)
                # h_n:     (num_layers, batch, hidden_size)
                context, h_n = self.gru(z[:, :-self.pred_steps, :], h_0)

                context = context[:, -1, :]
                # Use context vector for classification
                y_hat = self.classifier(context)

                loss = criterion(y_hat, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            if (epoch + 1) % args.disp_interval == 0:
                print(f'EPOCH: [{epoch + 1}/{args.epochs}] Loss: {np.mean(losses):.6f}')

    def evaluate(self, data_loader, args):
        self.encoder.eval()
        self.gru.eval()
        self.predictor.eval()
        self.classifier.eval()
        self.__freeze_parameters()

        predictions = []
        labels = []

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)

                # x: (batch, num_seq, channel, seq_len)
                (batch, num_seq, channel, seq_len) = x.shape
                x = x.view(batch * num_seq, channel, seq_len)
                z = self.encoder(x)
                z = z.view(batch, num_seq, self.feature_dim)  # (batch, num_seq, feature_dim)

                # Get context feature
                h_0 = self.gru.init_hidden(batch)
                # context: (batch, num_seq, hidden_size)
                # h_n:     (num_layers, batch, hidden_size)
                context, h_n = self.gru(z[:, :-self.pred_steps, :], h_0)

                context = context[:, -1, :]
                # Use context vector for classification
                y_hat = self.classifier(context)

                # TODO: test local representations
                # TODO: representation normalization and temperature

                labels.append(y.numpy()[:, -args.pred_steps - 1])
                predictions.append(y_hat.cpu().numpy())

        labels = np.concatenate(labels, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        predictions = np.argmax(predictions, axis=1)

        return get_performance(predictions, labels)

    def save(self):
        pass

    def load(self):
        pass

    def __repr__(self):
        return self.encoder.__repr__() + '\n' + self.gru.__repr__() + '\n' + self.predictor.__repr__() + \
               '\n' + self.classifier.__repr__()

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

        for p in self.gru.parameters():
            p.requires_grad = False

        for p in self.predictor.parameters():
            p.requires_grad = False
