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


def prepare_dataset(path, patients: Optional[Union[int, List]] = None):
    assert os.path.exists(path)
    file_names = os.listdir(path)

    data_list = []
    target_list = []

    if patients is None:
        candidate_files = file_names
    elif isinstance(patients, int):
        candidate_files = file_names[:patients]
    elif isinstance(patients, list):
        candidate_files = list(map(lambda p: os.path.join(path, p), patients))
    else:
        raise ValueError('Invalid patients param!')

    for filename in candidate_files:
        data = np.load(os.path.join(path, filename))
        data_list.append(
            np.concatenate(
                (data['eeg_fpz_cz'].reshape(-1, 1, data['eeg_fpz_cz'].shape[-1]),
                 data['eeg_pz_oz'].reshape(-1, 1, data['eeg_pz_oz'].shape[-1])),
                axis=1)
        )
        target_list.append(data['annotation'] - 1)

    # data_list = np.concatenate(data_list, axis=0)
    # target_list = np.concatenate(target_list, axis=0)

    # data = []
    # targets = []
    # for i in tqdm(range(0, len(data_list), stride)):
    #     if i + seq_len > len(data_list):
    #         break
    #     data.append(np.expand_dims(data_list[i: i + seq_len], axis=0))
    #     targets.append(np.expand_dims(target_list[i: i + seq_len], axis=0))
    # data = np.concatenate(data, axis=0)
    # targets = np.concatenate(targets, axis=0)

    return data_list, target_list


class SleepEDFDataset(Dataset):
    def __init__(self, x: List[np.ndarray], y: List[np.ndarray], seq_len, stride, return_label=False):
        self.return_label = return_label
        self.seq_len = seq_len
        self.stride = stride

        self.data = x
        self.targets = y

        length_list = []
        self.length_sum = 0
        for data_item in self.data:
            length_list.append((len(data_item) - self.seq_len) // self.stride + 1)
            self.length_sum += length_list[-1]
        self.length_list = np.cumsum(length_list)
        self.index_bucket = np.searchsorted(self.length_list, np.arange(self.length_sum), side='right')

    def __len__(self):
        return self.length_sum

    def __getitem__(self, item):
        bucket_id = self.index_bucket[item]
        offset = item - (0 if bucket_id == 0 else self.length_list[bucket_id - 1])

        if self.return_label:
            return (
                torch.from_numpy(
                    self.data[bucket_id][offset * self.stride:offset * self.stride + self.seq_len].astype(np.float32)),
                torch.from_numpy(
                    self.targets[bucket_id][offset * self.stride:offset * self.stride + self.seq_len].astype(np.long))
            )
        else:
            return torch.from_numpy(
                self.data[bucket_id][offset * self.stride:offset * self.stride + self.seq_len].astype(np.float32))

    def __repr__(self):
        return f"""
               ****************************************
               Model  : {self.__class__.__name__}
               Total Windows : {len(self)}
               Num Subjects : {len(self.data)}  
               Num Channels : {self.data[0].shape[1]}
               ****************************************
                """
