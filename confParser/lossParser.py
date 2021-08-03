# @Time : 2021/4/1 21:53
# @Author : Yangcheng Wu
# @FileName: lossParser.py
from loss.losses import *

lossMapper = {
    'softmax': Softmax,

}


def getloss(lossparams):
    return lossMapper[lossparams['type']](**lossparams)
