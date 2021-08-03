# @Time : 2021/4/28 0:50
# @Author : Yangcheng Wu
# @FileName: UTTEX.py
import torch
import torch.nn as nn
from model.UTTL import UTT
from utils.tools import quadra
import logging

class UTTEX(UTT):

    def __init__(self, params, type='111', diag=False):
        super(UTTEX, self).__init__(params)
        self.diag = diag
        logging.info('In UTTEX, the ut3type is {}, diag is {}'.format(type, diag))
        self.quadra_softmax = quadraS(512, 7323, type=type, diag=diag)
        self.quadra_softmax.models.data = params['loss.affine.weight'].squeeze()

    def forward(self, x):
        x = self.bn6(x)
        x = self.tdnn7(x)

        x = self.quadra_softmax(x)

        return x

    def pre_forward(self, x):
        x = self.bn6(x)
        x = self.tdnn7(x)
        return x

    def score(self, x, y):
        ans = torch.zeros(x.size()).float()
        x = self.pre_forward(x)
        y = self.pre_forward(y)
        P = self.quadra_softmax.within
        Q = self.quadra_softmax.between
        c = self.quadra_softmax.other
        if self.quadra_softmax.type[0] == '1':
            if self.diag:
                ans += self.onedMat(x, P, x) + self.onedMat(y, P, y)
            else:
                ans += quadra(x, P, x) + quadra(y, P, y)
        if self.quadra_softmax.type[1] == '1':
            if self.diag:
                ans += self.onedMat(x, Q, y)
            else:
                ans += 2 * quadra(x, Q, y)
        if self.quadra_softmax.type[2] == '1':
            ans += ((x + y) @ c).squeeze()
        return ans

    @staticmethod
    def onedMat(x, A, y):
        return (2 * (x * A).unsqueeze(1) @ y.unsqueeze(2)).squeeze()

class quadraS(nn.Module):

    def __init__(self, in_channel, out_channel, type='111', diag=False):
        super(quadraS, self).__init__()
        self.inc = in_channel
        self.out = out_channel
        self.diag = diag
        if not self.diag:
            self.within = nn.Parameter(torch.diag(torch.ones(in_channel)).float(), requires_grad=True)
            self.between = nn.Parameter(torch.diag(torch.ones(in_channel)).float(), requires_grad=True)
        else:
            self.within = nn.Parameter(torch.ones(in_channel).float(), requires_grad=True)
            self.between = nn.Parameter(torch.ones(in_channel).float(), requires_grad=True)
        self.other = nn.Parameter(torch.rand(self.inc, 1), requires_grad=True)
        self.models = nn.Parameter(torch.rand(self.out, self.inc), requires_grad=True)
        self.type = type

    def forward(self, x):
        '''
        :param x: batch * dimension
        :return: batch * spks
        '''
        batch = x.size(0)

        ans = torch.tensor(torch.zeros(batch, self.out).float()).cuda()

        if self.type[0] == '1':
            if self.diag:
                sx = x * x * self.within
                sm = self.models * self.models * self.within
            else:
                # batch
                sx = quadra(x, self.within, x)
                # spks
                sm = quadra(self.models, self.within, self.models)
            # batch * spk
            wit = sx.repeat(self.out, 1).transpose(0, 1) + sm.repeat(batch, 1)
            ans += wit

        if self.type[1] == '1':
            if self.diag:
                bet = 2 * x * self.between @ self.models.transpose(0, 1)
            else:
                # batch * spk
                bet = 2 * x @ self.between @ self.models.transpose(0, 1)
            ans += bet
        if self.type[2] == '1':
            # oned info
            oned = torch.tensor([])
            for i in range(x.size(0)):
                a = x[i, :].repeat(self.out, 1)
                b = a + self.models
                c = b @ self.other
                oned = torch.cat([oned, c.transpose(0, 1)], dim=0)
            ans += oned
        return ans


if __name__ == '__main__':

    A = quadraS(10, 20)
    a = torch.rand(5, 10)
    b = A(a)
    print(b.size())

    # a = torch.diag(torch.ones(4)).float()
    # inp.unsqueeze(0)
    # print(inp @ a @ inp)

    # inp = torch.rand(10, 512)
    # model = UTTEX('')
    # print(model(inp).size())