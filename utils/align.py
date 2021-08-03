# @Time : 2021/4/5 10:42
# @Author : Yangcheng Wu
# @FileName: align.py
import torch


def alignwithzeros(x: list):
    lengths = [i.size(1) for i in x]
    aligns = max(lengths)
    aligned = torch.stack([
        torch.cat([i, torch.zeros(i.size(0), aligns - i.size(1))], 1) for i in x
    ])
    return aligned, lengths

if __name__ == '__main__':
    a = torch.rand(10, 7)
    b = torch.rand(10, 9)
    c = torch.rand(10, 4)
    alignwithzeros([a, b, c])