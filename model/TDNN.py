# @Time : 2021/4/4 10:34
# @Author : Yangcheng Wu
# @FileName: TDNN.py
import torch.nn as nn
from model.basic_model import BasicModel, nonlinearMapper
from utils.align import alignwithzeros


class TDNN(BasicModel):

    def conv_bn_non_block(self, in_channel, out_channel, kernel_size, dilation):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      dilation=dilation,
                      padding=(kernel_size // 2) * dilation),
            nn.BatchNorm1d(out_channel),
            self.nonlinear,
        )

    def bn_non_block(self, in_channel):
        return nn.Sequential(
            nn.BatchNorm1d(in_channel),
            self.nonlinear,
        )

    def __init__(self, in_channel, spk_num, nonlinear='relu'):
        super(TDNN, self).__init__()

        self.nonlinear = nonlinearMapper[nonlinear]()
        self.frame1 = self.conv_bn_non_block(in_channel, 512, 5, 1)
        self.frame2 = self.conv_bn_non_block(512, 512, 3, 2)
        self.frame3 = self.conv_bn_non_block(512, 512, 3, 3)
        self.frame4 = self.conv_bn_non_block(512, 512, 1, 1)
        self.frame5 = self.conv_bn_non_block(512, 1500, 1, 1)

        self.utter1 = nn.Linear(3000, 512)
        self.non_bn1 = self.bn_non_block(512)
        # self.dropout1 = nn.Dropout()
        self.utter2 = nn.Linear(512, 512)
        self.non_bn2 = self.bn_non_block(512)
        # self.dropout2 = nn.Dropout()
        self.softmax = nn.Linear(512, spk_num)

        self._weight_init()

    @staticmethod
    def stat_pooling(x, dim=-1):
        return torch.cat([torch.mean(x, dim), torch.std(x, dim)], dim)

    @staticmethod
    def invariant_stat_pooling(x, lengths):
        '''
        I can't ensure if it is learnable, for extract only
        :param x:
        :param lengths:
        :return:
        '''

        lengths = torch.tensor(lengths).reshape(-1, 1)
        ans = torch.zeros(x.size(0), 2 * x.size(1)).float()
        for i, l in enumerate(lengths):
            tmpx = x[i, :, :l]
            mean = torch.sum(tmpx, -1) / l.float()
            std = torch.sqrt(
                torch.sum(
                    (tmpx - mean.repeat(tmpx.size(1), 1).transpose(0, 1)) ** 2
            , -1) / (l.float() - 1)
            )
            ans[i, :] = torch.cat([mean, std], -1)
        return ans

    def register(self, **named_parameters):
        for name, val in named_parameters.items():
            self.register_parameter(name, val)

    def frame_level(self, x):
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)
        return x

    def utter_level(self, x):
        x = self.utter1(x)
        x = self.non_bn1(x)
        x = self.utter2(x)
        x = self.non_bn2(x)
        return x

    def forward(self, x):
        x = self.frame_level(x)
        x = self.stat_pooling(x)
        x = self.utter_level(x)
        x = self.softmax(x)
        return x

    def extract(self, x, place=0):
        '''
        :param x: One input 2d feature, not batch
        :param place: 0 or 1
        :return: one xvector
        '''
        assert len(x.size()) == 2
        x = x.unsqueeze(0)
        x = self.frame_level(x)
        x = self.stat_pooling(x)
        x = self.utter1(x)
        if place == 0:
            return x
        elif place == 1:
            x = self.non_bn1(x)
            return self.utter2(x)
        else:
            raise Exception('place only accept 0 or 1')

    def batchextract(self, x, place=0):
        '''
        :param x: 3d batch features only
        :param lengths: length of each feature
        :param place: 0 or 1
        :return:
        '''
        batchedx, lengths = alignwithzeros(x)
        batchedx = self.frame_level(batchedx)
        x = self.invariant_stat_pooling(batchedx, lengths)
        x = self.utter1(x)
        if place == 0:
            return x
        elif place == 1:
            x = self.non_bn1(x)
            return self.utter2(x)
        else:
            raise Exception('`place` only accept 0 or 1')

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.)


class TOYTDNN(BasicModel):

    def conv_bn_non_block(self, in_channel, out_channel, kernel_size, dilation):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      dilation=dilation,
                      padding=(kernel_size // 2) * dilation),
            nn.BatchNorm1d(out_channel),
            self.nonlinear(),
        )

    def bn_non_block(self, in_channel):
        return nn.Sequential(
            nn.BatchNorm1d(in_channel),
            self.nonlinear(),
        )

    def __init__(self, in_channel, spk_num, nonlinear='relu'):
        super(TOYTDNN, self).__init__()
        self.nonlinear = nonlinearMapper[nonlinear]
        # self.tmp = nn.BatchNorm1d(in_channel)
        self.frame1 = self.conv_bn_non_block(in_channel, 100, 5, 1)

        self.utter1 = nn.Linear(200, 300)
        self.non_bn1 = self.bn_non_block(300)
        self.softmax = nn.Linear(300, spk_num)

        self._weight_init()

    @staticmethod
    def stat_pooling(x, dim=-1):
        return torch.cat([torch.mean(x, dim), torch.std(x, dim)], dim)

    @staticmethod
    def invariant_stat_pooling(x, lengths):
        '''
        I can't ensure if it is learnable, for extract only
        :param x:
        :param lengths:
        :return:
        '''
        lengths = torch.tensor(lengths).reshape(-1, 1)
        ans = torch.zeros(x.size(0), 2 * x.size(1)).float()
        for i, l in enumerate(lengths):
            tmpx = x[i, :, :l]
            mean = torch.sum(tmpx, -1) / l.float()
            std = torch.sqrt(
                torch.sum(
                    (tmpx - mean.repeat(tmpx.size(1), 1).transpose(0, 1)) ** 2
            , -1) / (l.float() - 1)
            )
            ans[i, :] = torch.cat([mean, std], -1)
        return ans

    def register(self, **named_parameters):
        for name, val in named_parameters.items():
            self.register_parameter(name, val)

    def frame_level(self, x):
        x = self.frame1(x)
        return x

    def utter_level(self, x):
        x = self.utter1(x)
        x = self.non_bn1(x)
        return x

    def forward(self, x):
        # x = self.tmp(x)
        x = self.frame_level(x)
        x = self.stat_pooling(x)
        x = self.utter_level(x)
        x = self.softmax(x)
        return x

    def extract(self, x, place=0):
        '''
        :param x: One input 2d feature, not batch
        :param place: 0 or 1
        :return: one xvector
        '''
        assert len(x.size()) == 2
        x = x.unsqueeze(0)
        x = self.frame_level(x)
        x = self.stat_pooling(x)
        x = self.utter1(x)
        return x

    def batchextract(self, x, place=0):
        '''
        :param x: 3d batch features only
        :param lengths: length of each feature
        :param place: 0 or 1
        :return:
        '''
        batchedx, lengths = alignwithzeros(x)
        batchedx = self.frame_level(batchedx)
        x = self.invariant_stat_pooling(batchedx, lengths)
        x = self.utter1(x)
        return x

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':

    import torch
    torch.manual_seed(19990110)
    a = TDNN(100, 100)


    b = torch.rand(100, 23)
    c = torch.rand(100, 23)
    d = torch.rand(100, 23)
    a.eval()
    ex1s = [a.extract(i, 1).squeeze() for i in [b, c, d]]
    ex2s = a.batchextract([b, c, d], 1)
    ex1s = torch.stack(ex1s)

    print(torch.sum((ex1s - ex2s) ** 2))
