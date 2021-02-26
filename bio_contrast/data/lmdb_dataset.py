"""
@Time    : 2020/12/7 16:36
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : lmdb_dataset.py
@Software: PyCharm
@Desc    : 
"""
import argparse
import glob
import os

import lmdb
import numpy as np
from torch.utils.data import Dataset
from tqdm.std import tqdm


class LmdbDataset(Dataset):
    def __init__(self, lmdb_file):
        self.lmdb_file = lmdb_file

        self.env = lmdb.open(lmdb_file, readonly=True, lock=False, readhead=False, meminit=False)

    def __getitem__(self, item):
        with self.env.begin(write=False) as txn:
            buffer = txn.get()
        data = np.frombuffer(buffer)

        return data

    def __len__(self):
        pass


def parse_args(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--dest-file', type=str, required=True)
    parser.add_argument('--commit-interval', type=int, default=100)

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


def create_lmdb_dataset(data_path, dest_file, commit_interval, suffix='npz'):
    assert dest_file.endswith('.lmdb')

    print('Start...')
    files = sorted(glob.glob(os.path.join(data_path, f'*.{suffix}')))
    file_size = np.load(files[0])['data'].nbytes
    # annotation = np.load(os.path.join(annotation_file))['annotation']
    dataset_size = file_size * len(files)
    print(f'Estimated dataset size: {dataset_size} bytes')

    env = lmdb.open(dest_file, map_size=dataset_size * 10)
    txn = env.begin(write=True)

    for idx, file in tqdm(enumerate(files), total=len(files), desc='Writing LMDB'):
        data = np.load(file)['data']
        key = os.path.basename(file).encode('ascii')
        txn.put(key, data)

        if (idx + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()


if __name__ == '__main__':
    args = parse_args()

    create_lmdb_dataset(args.data_path, args.dest_file, args.commit_interval)
