# @Time : 2021/5/3 14:05
# @Author : Yangcheng Wu
# @FileName: test.py
import torch.nn.functional as F
import torch

a = torch.arange(12).reshape(3, 4).float()
print(a)
b = torch.norm(a, dim=1)
print(b)
b = F.normalize(a)
print(b)