"""
@Time    : 2020/9/17 19:09
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : main.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import os
import pickle
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.std import tqdm

from bio_contrast.data import SLEEPEDF_SUBJECTS, ISRUC_SUBJECTS, SleepDataset
from bio_contrast.data import prepare_pretraining_dataset, prepare_evaluation_dataset
from bio_contrast.model import SleepContrast, SleepClassifier


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--data-category', type=str, default='sleepedf', choices=['sleepedf', 'isruc'])
    parser.add_argument('--save-path', type=str, default='./cache/checkpoints')
    # parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--rp', dest='relative_position', action='store_true')

    # parser.add_argument('--num-patient', type=int, default=5)
    parser.add_argument('--seq-len', type=int, default=10)
    parser.add_argument('--input-channels', type=int, default=2)
    parser.add_argument('--hidden-channels', type=int, default=16)
    # parser.add_argument('--num-seq', type=int, default=20)
    parser.add_argument('--pred-steps', type=int, default=5)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--finetune-ratio', type=float, nargs='+', default=[0.01, 0.5, 0.1, 0.2, 0.3, 0.4,
                                                                            0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-lr', dest='finetune_lr', type=float, default=1e-3)

    args_parsed = parser.parse_args()

    if verbose:
        message = ''
        message += '-------------------------------- Args ------------------------------\n'
        for k, v in sorted(vars(args_parsed).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------------------- End ----------------------------------'
        print(message)

    return args_parsed


def compute_cpc_targets(batch_size, pred_steps, seq_len):
    targets = torch.zeros(batch_size, pred_steps, seq_len, batch_size).long()
    for i in range(batch_size):
        for j in range(pred_steps):
            targets[i, j, seq_len - pred_steps + j, i] = 1

    targets = targets.cuda()
    targets = targets.view(batch_size * pred_steps, seq_len * batch_size)
    targets = targets.argmax(dim=1)

    return targets


def compute_position_targets(batch_size, seq_len):
    targets = torch.zeros(batch_size, seq_len, seq_len, batch_size).long()
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(seq_len):
                targets[i, j, k, i] = 1

    targets = targets.cuda()
    targets = targets.view(batch_size * seq_len, seq_len * batch_size)
    targets = targets.argmax(dim=1)

    return targets


def pretrain(model, train_loader, split_id, args):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98),
                           eps=1e-09, weight_decay=1e-4, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    cpc_targets = compute_cpc_targets(args.batch_size, args.pred_steps, args.seq_len)
    if args.relative_position:
        position_targets = compute_position_targets(args.batch_size, args.seq_len)

    model.train()
    for epoch in range(args.epochs):
        loss_list = []

        for x, y in tqdm(train_loader, desc=f'EPOCH: [{epoch + 1}/{args.epochs}]'):
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            if args.relative_position:
                cpc_score, rp_score = model(x)
                loss = criterion(cpc_score, cpc_targets) + criterion(rp_score, position_targets)
            else:
                score = model(x)
                loss = criterion(score, cpc_targets)

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        print(f'Loss: {np.mean(loss_list)}')

        if (epoch + 1) % 10 == 0:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            torch.save(model.state_dict(), os.path.join(args.save_path, f'encoder_{split_id}_epoch_{epoch}.pth'))


def finetune(classifier, finetune_loader, args):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()),
                           lr=args.finetune_lr, betas=(0.9, 0.98), eps=1e-09,
                           weight_decay=1e-4, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    classifier.train()

    for epoch in range(args.finetune_epochs):
        losses = []
        for x, y in tqdm(finetune_loader, desc=f'EPOCH:[{epoch + 1}/{args.finetune_epochs}]'):
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            y_hat = classifier(x)
            loss = criterion(y_hat, y[:, -args.pred_steps - 1])

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        print(f'Loss: {np.mean(losses)}')


def evaluate(classifier, test_loader, args):
    classifier.eval()

    predictions = []
    labels = []
    for x, y in tqdm(test_loader):
        x = x.cuda()

        with torch.no_grad():
            y_hat = classifier(x)

        labels.append(y.numpy()[:, -args.pred_steps - 1])
        predictions.append(y_hat.cpu().numpy())

    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')

    return {'accuracy': accuracy, 'f1_micro': f1_micro, 'f1_macro': f1_macro}


def setup_seed(seed):
    warnings.warn(f'You have chosen to seed ({seed}) training. '
                  f'This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! '
                  f'You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = parse_args()

    # setup_seed(args.seed)

    results = {}
    if args.data_category == 'sleepedf':
        all_subjects = SLEEPEDF_SUBJECTS
    else:
        all_subjects = ISRUC_SUBJECTS

    if args.data_path is None:
        warnings.warn('The data path is not specified, using default setting...')
        if args.data_category == 'sleepedf':
            args.data_path = './data/sleepedf'
        else:
            args.data_path = './data/ISRUC-SLEEP/subgroup3'

    for i in range(len(all_subjects)):
        train_subjects = list(set(all_subjects) - {all_subjects[i]})
        test_subjects = [all_subjects[i]]

        print(f'**********ROUND {i + 1} STARTED**********')
        print('Train subjects:', train_subjects)
        print('Test subjects:', test_subjects)

        pretrain_data, pretrain_targets = prepare_pretraining_dataset(args.data_path, seq_len=args.seq_len,
                                                                      data_category=args.data_category,
                                                                      patients=train_subjects)

        pretrain_dataset = SleepDataset(pretrain_data, pretrain_targets, return_label=True)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size,
                                     drop_last=True, shuffle=True, pin_memory=True)

        model = SleepContrast(input_channels=args.input_channels, hidden_channels=args.hidden_channels,
                              feature_dim=args.feature_dim, pred_steps=args.pred_steps,
                              batch_size=args.batch_size, num_seq=args.seq_len,
                              relative_position=args.relative_position, kernel_sizes=[7, 11, 7])
        model.cuda()

        pretrain(model, pretrain_loader, i, args)

        current_split_result = {}
        for ratio in tqdm(args.finetune_ratio):
            print(f'Test finetune ratio {ratio}...')
            classifier = SleepClassifier(input_channels=args.input_channels, hidden_channels=args.hidden_channels,
                                         num_classes=args.num_classes, feature_dim=args.feature_dim,
                                         pred_steps=args.pred_steps, batch_size=args.batch_size,
                                         num_seq=args.seq_len, kernel_sizes=[7, 11, 7])
            classifier.cuda()

            # Copying encoder params
            for finetune_param, pretraining_param in zip(classifier.encoder.parameters(), model.encoder.parameters()):
                finetune_param.data = pretraining_param.data

            # Copying gru params
            for finetune_param, pretraining_param in zip(classifier.gru.parameters(), model.gru.parameters()):
                finetune_param.data = pretraining_param.data

            classifier.freeze_parameters()

            finetune_data, finetune_targets = prepare_evaluation_dataset(args.data_path,
                                                                         data_category=args.data_category,
                                                                         seq_len=args.seq_len,
                                                                         patients=train_subjects,
                                                                         sample_ratio=ratio)
            finetune_dataset = SleepDataset(finetune_data, finetune_targets, return_label=True)
            finetune_loader = DataLoader(finetune_dataset, batch_size=args.batch_size,
                                         drop_last=True, shuffle=True, pin_memory=True)

            finetune(classifier, finetune_loader, args)
            torch.save(classifier.state_dict(), os.path.join(args.save_path, f'classifier_{i}_{ratio}.pth'))

            test_data, test_targets = prepare_evaluation_dataset(args.data_path,
                                                                 data_category=args.data_category,
                                                                 seq_len=args.seq_len,
                                                                 patients=test_subjects,
                                                                 sample_ratio=1.0)
            test_dataset = SleepDataset(test_data, test_targets, return_label=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     drop_last=True, shuffle=True, pin_memory=True)
            current_ratio_result = evaluate(classifier, test_loader, args)
            current_split_result[ratio] = current_ratio_result
        results[i] = current_split_result

    print(results)
    with open(os.path.join(args.save_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
