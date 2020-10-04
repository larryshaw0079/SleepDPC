"""
@Time    : 2020/9/26 16:45
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : test_model.py
@Software: PyCharm
@Desc    : 
"""
import os
import sys
import argparse

import numpy as np

from torchsummary import summary
import torch
import torch.nn as nn

sys.path.append('./')

from DPC.dpc.model_3d import DPC_RNN


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', default='resnet18', type=str)
    parser.add_argument('--model', default='dpc-rnn', type=str)
    parser.add_argument('--dataset', default='ucf101', type=str)
    parser.add_argument('--seq_len', default=5, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=8, type=int, help='number of video blocks')
    parser.add_argument('--pred_step', default=3, type=int)
    parser.add_argument('--ds', default=3, type=int, help='frame downsampling rate')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default='0,1', type=str)
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
    parser.add_argument('--prefix', default='tmp', type=str, help='prefix of checkpoint filename')
    parser.add_argument('--train_what', default='all', type=str)
    parser.add_argument('--img_dim', default=128, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    print(args)

    model = DPC_RNN(sample_size=args.img_dim,
                        num_seq=args.num_seq,
                        seq_len=args.seq_len,
                        network=args.net,
                        pred_step=args.pred_step)



    # print(model)
    device = 'cuda:0'
    model = model.to(device)
    summary(model, input_size=(args.num_seq, 3, args.seq_len, args.img_dim, args.img_dim), device='cuda')

    batch_data = torch.randn(args.batch_size, args.num_seq, 3, args.seq_len, args.img_dim, args.img_dim)

    batch_data = batch_data.to(device)

    scores, mask = model(batch_data)

    print(f'Input: {batch_data.shape}')
    print(f'Score: {scores.shape}')
    print(f'Mask: {mask.shape}')
