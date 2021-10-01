"""
@Time    : 2021/10/1 17:15
@File    : data.py
@Software: PyCharm
@Desc    : 
"""
import os
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import Dataset
from tqdm.std import tqdm

EPS = 1e-8


def tackle_denominator(x: np.ndarray):
    x[x == 0.0] = EPS
    return x


def tensor_standardize(x: np.ndarray, dim=-1):
    x_mean = np.expand_dims(x.mean(axis=dim), axis=dim)
    x_std = np.expand_dims(x.std(axis=dim), axis=dim)
    return (x - x_mean) / tackle_denominator(x_std)


class SleepDataset(Dataset):
    def __init__(self, data_path, data_name, num_epoch, patients: List = None, preprocessing: str = 'none', modal='eeg',
                 return_idx=False, verbose=True):
        assert isinstance(patients, list)

        self.data_path = data_path
        self.data_name = data_name
        self.patients = patients
        self.preprocessing = preprocessing
        self.modal = modal
        self.return_idx = return_idx

        assert preprocessing in ['none', 'quantile', 'standard']
        assert modal in ['eeg', 'emg', 'eog']

        self.data = []
        self.labels = []

        for i, patient in enumerate(tqdm(patients, desc='::: LOADING EEG DATA :::')):
            # if verbose:
            #     print(f'[INFO] Processing the {i + 1}-th patient {patient}...')
            data = np.load(os.path.join(data_path, patient))
            if data_name == 'sleepedf':
                if modal == 'eeg':
                    recordings = np.stack([data['eeg_fpz_cz'], data['eeg_pz_oz']], axis=1)
                elif modal == 'emg':
                    recordings = np.expand_dims(data['emg'], axis=1)
                elif modal == 'eog':
                    recordings = np.expand_dims(data['eog'], axis=1)
                else:
                    raise ValueError

                annotations = data['annotation']
            elif data_name == 'isruc':
                recordings = np.stack([data['F3_A2'], data['C3_A2'], data['F4_A1'], data['C4_A1'],
                                       data['O1_A2'], data['O2_A1']], axis=1)
                annotations = data['stage_label'].flatten()
            else:
                raise ValueError

            if preprocessing == 'standard':
                # print(f'[INFO] Applying standard scaler...')
                # scaler = StandardScaler()
                # recordings_old = recordings
                # recordings = []
                # for j in range(recordings_old.shape[0]):
                #     recordings.append(scaler.fit_transform(recordings_old[j].transpose()).transpose())
                # recordings = np.stack(recordings, axis=0)

                recordings = tensor_standardize(recordings, dim=-1)
            elif preprocessing == 'quantile':
                # print(f'[INFO] Applying quantile scaler...')
                scaler = QuantileTransformer(output_distribution='normal')
                recordings_old = recordings
                recordings = []
                for j in range(recordings_old.shape[0]):
                    recordings.append(scaler.fit_transform(recordings_old[j].transpose()).transpose())
                recordings = np.stack(recordings, axis=0)
            else:
                # print(f'[INFO] Convert the unit from V to uV...')
                recordings *= 1e6

            # if verbose:
            #     print(f'[INFO] The shape of the {i + 1}-th patient: {recordings.shape}...')
            recordings = recordings[:(recordings.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch,
                                                                                             *recordings.shape[1:])
            annotations = annotations[:(annotations.shape[0] // num_epoch) * num_epoch].reshape(-1, num_epoch)

            assert recordings.shape[:2] == annotations.shape[:2]

            self.data.append(recordings)
            self.labels.append(annotations)

        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        self.idx = np.arange(self.data.shape[0] * self.data.shape[1]).reshape(-1, self.data.shape[1])
        self.full_shape = self.data[0].shape

    def __getitem__(self, item):
        x = self.data[item]
        y = self.labels[item]

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))

        if self.return_idx:
            return x, y, torch.from_numpy(self.idx[item].astype(np.long))
        else:
            return x, y

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return """
**********************************************************************
Dataset Summary:
Preprocessing: {}
# Instance: {}
Shape of an Instance: {}
Selected patients: {}
**********************************************************************
            """.format(self.preprocessing, len(self.data), self.full_shape, self.patients)
