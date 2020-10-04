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


class SleepEDFDataset(Dataset):
    def __init__(self, path, seq_len, stride=1, patients=None, return_label=False):
        self.return_label = return_label
        self.seq_len = seq_len

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

        self.data = []
        self.targets = []
        for i in tqdm(range(0, len(candidate_data), stride)):
            if (i + seq_len > len(candidate_data)):
                break
            self.data.append(np.expand_dims(candidate_data[i: i + seq_len], axis=0))
            self.targets.append(np.expand_dims(candidate_target[i: i + seq_len], axis=0))
        self.data = np.concatenate(self.data, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

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
