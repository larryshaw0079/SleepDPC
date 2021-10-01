"""
@Time    : 2020/11/28 13:21
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : data_process.py
@Software: PyCharm
@Desc    : 
"""

import os
import warnings

import numpy as np
from scipy import signal
from tqdm.std import tqdm

DATA_PATH = '../data/sleepedf'
DEST_PATH = './data/sleepedf_processed'
MAX_PERTURBATION = 5 * 100
NUM_SAMPLING = 5
WINDOW_SIZE = 1024
SAMPLING_RATE = 100

if __name__ == '__main__':
    if not os.path.exists(DEST_PATH):
        warnings.warn(f'The path {DEST_PATH} does not existed, created.')
        os.makedirs(DEST_PATH)

    file_list = os.listdir(DATA_PATH)
    for file in file_list:
        print(f'Processing file {file}...')
        file_name = os.path.join(DATA_PATH, file)
        file_prefix = file.split('.')[0]

        if not os.path.exists(os.path.join(DEST_PATH, file_prefix, 'time')):
            os.makedirs(os.path.join(DEST_PATH, file_prefix, 'time'))
        if not os.path.exists(os.path.join(DEST_PATH, file_prefix, 'frequency')):
            os.makedirs(os.path.join(DEST_PATH, file_prefix, 'frequency'))

        data = np.load(file_name)

        # recordings = np.stack([data['eeg_fpz_cz'], data['eeg_pz_oz']], axis=1)
        recording_ch1 = data['eeg_fpz_cz']  # (num_epoch, epoch_length)
        recording_ch2 = data['eeg_pz_oz']
        annotations = data['annotation']
        num_epochs, epoch_length = recording_ch1.shape
        recording_ch1 = recording_ch1.reshape(-1)
        recording_ch2 = recording_ch2.reshape(-1)

        for idx in tqdm(range(num_epochs)):
            time_q = np.stack([recording_ch1[idx * epoch_length:(idx + 1) * epoch_length],
                               recording_ch2[idx * epoch_length:(idx + 1) * epoch_length]], axis=0).astype(np.float32)
            # label = annotations[idx]

            _, freq_q = signal.welch(time_q, SAMPLING_RATE, window='hamming', nperseg=WINDOW_SIZE)

            time_k_list = []
            freq_k_list = []

            candidates = np.concatenate([np.arange(idx * epoch_length - MAX_PERTURBATION, idx),
                                         np.arange(idx + 1, idx * epoch_length + MAX_PERTURBATION + 1)])
            candidates = np.clip(candidates, a_min=0, a_max=(num_epochs - 1) * epoch_length)
            indices = np.random.choice(candidates, size=NUM_SAMPLING, replace=False)

            for i_sample in range(NUM_SAMPLING):
                data_k = np.stack([recording_ch1[indices[i_sample]:indices[i_sample] + epoch_length],
                                   recording_ch2[indices[i_sample]:indices[i_sample] + epoch_length]], axis=0)

                time_k_list.append(data_k)
                freq_k_list.append(signal.welch(data_k, SAMPLING_RATE, window='hamming', nperseg=WINDOW_SIZE)[1])

            time_k = np.stack(time_k_list, axis=0).astype(np.float32)
            freq_k = np.stack(freq_k_list, axis=0).astype(np.float32)

            time_data = np.concatenate([np.expand_dims(time_q, axis=0), time_k], axis=0)
            freq_data = np.concatenate([np.expand_dims(freq_q, axis=0), freq_k], axis=0)

            np.savez(os.path.join(DEST_PATH, file_prefix, 'time', f'{idx}.npz'), data=time_data)
            np.savez(os.path.join(DEST_PATH, file_prefix, 'frequency', f'{idx}.npz'), data=freq_data)
        np.savez(os.path.join(DEST_PATH, file_prefix, 'annotation.npz'), annotation=annotations)

        # with open(os.path.join(DEST_PATH, file_prefix, f'T-{idx}.pkl'), 'wb') as f:
        #     pickle.dump(epoch_time, f)
        #
        # with open(os.path.join(DEST_PATH, file_prefix, f'F-{idx}.pkl'), 'wb') as f:
        #     pickle.dump(epoch_freq, f)
