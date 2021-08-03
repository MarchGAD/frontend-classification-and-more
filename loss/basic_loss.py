# @Time : 2021/4/4 8:21
# @Author : Yangcheng Wu
# @FileName: basic_model.py.py
from abc import ABCMeta, abstractmethod


class BasicLoss(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self.lparams = {}
        self.params = {}

    @abstractmethod
    def __create_leranable_params__(self):
        pass

    @abstractmethod
    def loss_func(self):
        pass
