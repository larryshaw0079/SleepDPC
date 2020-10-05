"""
@Time    : 2020/9/17 19:09
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import os

import numpy as np
from tqdm.std import tqdm

import torch
from torch.utils.data import Dataset


def prepare_dataset(path, seq_len, stride=1, patients=None):
    assert os.path.exists(path)
    file_names = os.listdir(path)

    candidate_data = []
    candidate_target = []

    for filename in file_names[: len(file_names) if patients is None else patients]:
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
        if (i + seq_len > len(candidate_data)):
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
