"""
@Time    : 2020/9/17 19:13
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : test_dataset.py
@Software: PyCharm
@Desc    : 
"""
from bio_contrast.data import SleepEDFDataset


def test_sleep_edfdataset():
    dataset = SleepEDFDataset('./data/sleepedf', patient=0)
    print(dataset)
