"""
@Time    : 2020/11/9 16:52
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : run.py
@Software: PyCharm
@Desc    : 
"""

import argparse

from bio_contrast.util import setup_seed


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--data-category', type=str, default='sleepedf', choices=['sleepedf', 'bciiv'])
    parser.add_argument('--downstream-task', type=str, default='sleep', choices=['sleep', 'bci', 'emotion'])

    # Model
    parser.add_argument('--backend', type=str, default='dpc', choices=['moco', 'dcc', 'dpc', 'coclr'])

    # Training
    parser.add_argument('pretrain-epochs', type=int, default=200)
    parser.add_argument('finetune-epochs', type=int, default=10)

    # Optimization
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-3)
    parser.add_argument('--wd', dest='weight_decay', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)

    # Misc
    parser.add_argument('--disp-interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None)

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


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        setup_seed(args.seed)
