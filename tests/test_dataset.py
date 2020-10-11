"""
@Time    : 2020/9/17 19:13
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : test_dataset.py
@Software: PyCharm
@Desc    : 
"""

from torch.utils.data import DataLoader

from bio_contrast.data import prepare_sleepedf_dataset, SleepEDFDataset


def test_sleep_edfdataset():
    data, targets = prepare_sleepedf_dataset(path='./data/sleepedf', patients=2)
    print(len(data))
    print(len(targets))
    print(data[0].shape, data[1].shape)
    dataset = SleepEDFDataset(data, targets, seq_len=20, stride=1, return_label=True)
    data_loader = DataLoader(dataset, batch_size=128, drop_last=True, shuffle=True, pin_memory=True)

    # for x, y in tqdm(data_loader):
    #     print(x.shape)
    #     print(y.shape)

    print(dataset)
    print(dataset.length_list)
    print(dataset.index_bucket)

    for i in range(len(dataset)):
        print(f'{dataset[i][0].shape} - {dataset[i][1].shape}')
