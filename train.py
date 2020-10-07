"""
@Time    : 2020/10/5 11:46
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : train.py
@Software: PyCharm
@Desc    : 
"""
import os
import random
import argparse
import warnings

import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from rich.progress import track

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.model_selection import train_test_split

from bio_contrast.data import prepare_dataset, SleepEDFDataset
from bio_contrast.model import SleepContrast


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data-path', type=str, default='./data/sleepedf')
    parser.add_argument('--save-path', type=str, default='./cache/checkpoints')
    parser.add_argument('--save-freq', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2020)

    # Distributed training
    parser.add_argument('--world-size', type=int, default=4)
    parser.add_argument('--master-port', type=str, default='12355')
    # parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument('--dist-backend', type=str, default='nccl')

    # Model params
    parser.add_argument('--num-patient', type=int, default=5)
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--input-channels', type=int, default=2)
    parser.add_argument('--hidden-channels', type=int, default=16)
    parser.add_argument('--num-seq', type=int, default=20)
    parser.add_argument('--pred-steps', type=int, default=5)
    parser.add_argument('--feature-dim', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--finetune-ratio', type=float, default=0.1)

    # Training params
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr-step', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train-ratio', type=float, default=0.7)

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


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    warnings.warn(f'You have chosen to seed ({seed}) training. '
                  f'This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! '
                  f'You may see unexpected behavior when restarting '
                  f'from checkpoints.')


def save_checkpoint(state, filename):
    torch.save(state, filename)


def compute_targets(args, device=0):
    targets = torch.zeros(args.batch_size, args.pred_steps, args.num_seq, args.batch_size).long()
    for i in range(args.batch_size):
        for j in range(args.pred_steps):
            targets[i, j, args.num_seq - args.pred_steps + j, i] = 1

    targets = targets.cuda(device)
    targets = targets.view(args.batch_size * args.pred_steps, args.num_seq * args.batch_size)
    targets = targets.argmax(dim=1)
    return targets


def worker(rank, world_size, args):
    print(f'[INFO ({rank})] Process started.')

    dist.init_process_group(backend=args.dist_backend, init_method=f'tcp://127.0.0.1:{args.master_port}',
                            world_size=world_size, rank=rank)

    torch.cuda.set_device(rank)

    model = SleepContrast(input_channels=args.input_channels, hidden_channels=args.hidden_channels,
                          feature_dim=args.feature_dim, pred_steps=args.pred_steps,
                          batch_size=args.batch_size, num_seq=args.num_seq, kernel_sizes=[7, 11, 7])
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98),
                           eps=1e-09, weight_decay=1e-4, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    criterion = nn.CrossEntropyLoss().cuda(rank)
    targets = compute_targets(args, device=rank)

    data_x, data_y = prepare_dataset(path=args.data_path, patients=args.num_patient)

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size=args.train_ratio)

    train_dataset = SleepEDFDataset(train_x, train_y, seq_len=args.seq_len, stride=args.stride, return_label=True)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              drop_last=True, shuffle=(train_sampler is None), pin_memory=True)

    print(f'[INFO ({rank})] Start training...')
    model.train()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        print(f'EPOCH [{epoch + 1}/{args.epochs}] started.')

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            score = model(x)
            loss = criterion(score, targets)

            loss.backward()
            optimizer.step()

        scheduler.step()
        if rank == 0 and (epoch+1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_path, f'pretrain_epoch_{epoch}_seed_{args.seed}.path.tar'))
        print('finished.')


if __name__ == '__main__':
    args = parse_args()

    setup_seed(args.seed)

    mp.spawn(worker, nprocs=args.world_size, args=(args.world_size, args))
