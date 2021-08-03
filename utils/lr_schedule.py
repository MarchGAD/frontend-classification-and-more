# @Time : 2021/4/14 13:47
# @Author : Yangcheng Wu
# @FileName: lr_schedule.py
import logging

class LRS:

    def __init__(self):
        self.valaccs = []
        self.vallosses = []
        self.testaccs = []
        self.testlosses = []

    def add(self, valacc, valloss, testacc, testloss):
        self.valaccs.append(valacc)
        self.vallosses.append(valloss)
        self.testaccs.append(testacc)
        self.testlosses.append(testloss)

    def half_scuedule(self, num, opt):
        if len(self.vallosses) > num:
            if self.vallosses[-1] > min(self.vallosses[(-num - 1):-1]):
                lrs = self.half(opt)
                logging.info('Haven\'t imporve for {} epochs, reduce lr to {}'.
                             format(num, str(lrs)))

    def half(self, opt):
        lrs = []
        for param in opt.param_groups:
            param['lr'] /= 2.0
            lrs.append(param['lr'])
        return lrs

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    mod = nn.Linear(10, 2)
    a = torch.optim.SGD(mod.parameters(), lr=1)
    print(a)
    for t in a.param_groups:
        print(t['lr'])
    b = LRS()
    b.half(a)
    for t in a.param_groups:
        print(t['lr'])