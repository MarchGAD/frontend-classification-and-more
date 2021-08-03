# @Time : 2021/4/4 10:32
# @Author : Yangcheng Wu
# @FileName: basic_model.py.py
from abc import ABCMeta, abstractmethod
import torch.nn as nn

nonlinearMapper = {
    'relu': nn.ReLU,
    'prelu': nn.PReLU,
    'lrelu': nn.LeakyReLU,

}


class BasicModel(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def register(self, params):
        pass

    @abstractmethod
    def _weight_init(self):
        pass

    @abstractmethod
    def extract(self):
        pass