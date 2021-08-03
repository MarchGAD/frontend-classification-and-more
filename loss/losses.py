# @Time : 2021/4/4 8:21
# @Author : Yangcheng Wu
# @FileName: losses.py
import torch
from loss.basic_loss import BasicLoss
import torch.nn as nn
import torch.nn.functional as F


class Softmax(BasicLoss):

    def __init__(self, **lossparams):
        super(Softmax, self).__init__()

    def __create_leranable_params__(self):
        pass

    def loss_func(self, res, label):
        '''
        totally the same as torch.nn.CrossEntrophyLoss
        :param res:
        :param label: 1d
        :return:
        '''
        soft = F.softmax(res, dim=1)
        logsoft = -torch.log(soft)
        selected = logsoft[[i for i in range(len(label))], label]
        ans = torch.sum(selected) / res.size(0)
        return ans



if __name__ == '__main__':
    # b = torch.arange(10).reshape(2, 5).float()
    # l = torch.tensor([1, 3])
    #
    # ans = a.loss_func(b, l)
    # print(ans)
    a = Softmax()
    t = nn.CrossEntropyLoss()
    # print(t(b, l))
    torch.manual_seed(114514)
    inp = torch.rand(10, 4)
    label = torch.LongTensor(10).random_() % 5
    model = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(10, 15),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(15, 5)
    )


    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    for i in range(3):
        opt.zero_grad()
        ans = model(inp)
        # loss = t(ans, label)
        loss = a.loss_func(ans, label)
        print(loss.item())
        loss.backward()
        opt.step()