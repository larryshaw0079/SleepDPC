# Self-Supervised Learning for Sleep Stage Classification with Predictive and Discriminative Contrastive Coding

This repository contains the implementation of our proposed model `SleepDPC` of paper *Self-Supervised Learning for
Sleep Stage Classification with Predictive and Discriminative Contrastive Coding* in ICASSP2021.

![](https://i.loli.net/2021/10/01/sBgdmz4CHfObIZL.png)

# Basic Usage

A typical command to run the model on the SleepEDF dataset would be:

```bash
python main.py --data-name sleepedf --data-path <your-data-path> --pretrain-epochs 50 --seed 2020 --optimizer adam --fold 0 --kfold 10 --batch-size 32 --channels 2
```

To see more options, please type `python main.py -h`.

# Main Results

![](https://i.loli.net/2021/10/01/veIyYGi7FEauQ9Z.png)

# Citation

If you find our paper is helpful for your research, please cite this paper:

```latex
@INPROCEEDINGS{9414752,
  author={Xiao, Qinfeng and Wang, Jing and Ye, Jianan and Zhang, Hongjun and Bu, Yuyan and Zhang, Yiqiong and Wu, Hao},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Self-Supervised Learning for Sleep Stage Classification with Predictive and Discriminative Contrastive Coding}, 
  year={2021},
  volume={},
  number={},
  pages={1290-1294},
  doi={10.1109/ICASSP39728.2021.9414752}
}
```

