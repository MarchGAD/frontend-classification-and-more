# @Time : 2021/4/6 8:34
# @Author : Yangcheng Wu
# @FileName: collate_fns.py
'''
This specifies the collate_fn in torch.utils.data.DataLoader, which organizes each mini-batch in training.
'''

import torch.utils.data as Data
from torch.utils.data._utils.collate import default_collate
import torch
import random


def min_align_collate(batch_data):
    lens = [i[0].size(1) for i in batch_data]
    minl = min(lens)
    return torch.stack([i[0][:, :minl] for i in batch_data]), torch.LongTensor([i[1] for i in batch_data])


def invariant_collate(batch_data):
    return batch_data


def invariant_align_collate(batch_data):
    lengths = [len(i) for i in batch_data]
    align = max(lengths)
    ori = torch.zeros(len(lengths), align)
    for i, j in enumerate(batch_data):
        ori[i, :lengths[i]] = j
    return ori


class ParameterCollate:

    def __init__(self):
        self.min_len = 200
        self.max_len = 400
        self.len = 400

    def fix_crop(self, batch_data):
        datalist = []
        labellist = []
        for data, label in batch_data:
            idx = random.randint(0, data.size(1) - self.len)
            datalist.append(data[:, idx:idx + self.len])
            labellist.append(label)
        return torch.stack(datalist), torch.LongTensor(labellist)


param_collate = ParameterCollate()

if __name__ == '__main__':
    a = torch.rand(5, 10)
    b = torch.rand(5, 2)
    c = torch.rand(5, 7)
    print(min_align_collate([a, b, c]).size())



