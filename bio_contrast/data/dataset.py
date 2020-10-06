"""
@Time    : 2020/9/17 19:09
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import os
import warnings
from typing import Optional, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.std import tqdm

SELECTED_SUBJECTS = ['SC4031E0.npz',
                     'SC4641E0.npz',
                     'SC4092E0.npz',
                     'SC4022E0.npz',
                     'SC4202E0.npz',
                     'SC4582G0.npz',
                     'SC4621E0.npz',
                     'SC4502E0.npz',
                     'SC4271F0.npz',
                     'SC4311E0.npz',
                     'SC4432E0.npz',
                     'SC4122E0.npz',
                     'SC4382F0.npz',
                     'SC4332F0.npz',
                     'SC4192E0.npz',
                     'SC4421E0.npz',
                     'SC4622E0.npz',
                     'SC4611E0.npz',
                     'SC4711E0.npz',
                     'SC4442E0.npz']


def prepare_dataset(path, seq_len, stride=1, in_memory=True, patients: Optional[Union[int, List]] = None):
    assert os.path.exists(path)
    file_names = os.listdir(path)

    candidate_data = []
    candidate_target = []

    if patients is None:
        candidate_files = file_names
    elif isinstance(patients, int):
        candidate_files = file_names[:patients]
    elif isinstance(patients, list):
        candidate_files = list(map(lambda p : os.path.join(path, p), patients))
    else:
        raise ValueError('Invalid patients param!')

    for filename in candidate_files:
        data = np.load(os.path.join(path, filename))
        candidate_data.append(
            np.concatenate(
                (data['eeg_fpz_cz'].reshape(-1, 1, data['eeg_fpz_cz'].shape[-1]),
                 data['eeg_pz_oz'].reshape(-1, 1, data['eeg_pz_oz'].shape[-1])),
                axis=1)
        )
        candidate_target.append(data['annotation'] - 1)


    candidate_data = np.concatenate(candidate_data, axis=0)
    candidate_target = np.concatenate(candidate_target, axis=0)

    data = []
    targets = []
    for i in tqdm(range(0, len(candidate_data), stride)):
        if i + seq_len > len(candidate_data):
            break
        data.append(np.expand_dims(candidate_data[i: i + seq_len], axis=0))
        targets.append(np.expand_dims(candidate_target[i: i + seq_len], axis=0))
    data = np.concatenate(data, axis=0)
    targets = np.concatenate(targets, axis=0)

    return data, targets


class SleepEDFDataset(Dataset):
    def __init__(self, x, y, return_label=False):
        self.return_label = return_label

        self.data = x
        self.targets = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.return_label:
            return (
                torch.from_numpy(self.data[item].astype(np.float32)),
                torch.from_numpy(self.targets[item].astype(np.long))
            )
        else:
            return torch.from_numpy(self.data[item].astype(np.float32))

    def __repr__(self):
        return f"""
               ****************************************
               Model  : {self.__class__.__name__}
               Length : {len(self)}
               ****************************************
                """
