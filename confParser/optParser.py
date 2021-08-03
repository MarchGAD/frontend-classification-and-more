# @Time : 2021/4/8 15:48
# @Author : Yangcheng Wu
# @FileName: optParser.py
import torch.optim as opt

optimizerMapper = {
    'sgd': opt.SGD,
    'adam': opt.Adam,
    'adadelta': opt.Adadelta
}


def getoptimizer(optparams, model):
    if optparams['type'] == 'adam':
        return optimizerMapper[optparams['type']](
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optparams['lr'],
            weight_decay=optparams['weight_decay'],
            eps=optparams['eps']
        )
    else:
        return optimizerMapper[optparams['type']](
            filter(lambda p:p.requires_grad, model.parameters()),
            lr=optparams['lr'],
            weight_decay=optparams['weight_decay']
        )