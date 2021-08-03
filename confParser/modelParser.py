# @Time : 2021/4/1 21:53
# @Author : Yangcheng Wu
# @FileName: modelParser.py
import torch
from model.TDNN import TDNN, TOYTDNN
from model.UTTL import UTT
from model.UTTEX import UTTEX


def getmodel(params, spk_num):
    name = params['type']
    if name == 'tdnn':
        return TDNN(
            in_channel=params['in_channel'],
            spk_num=spk_num,
            nonlinear=params['nonlinear'],
        )
    elif name == 'toy':
        return TOYTDNN(
            in_channel=params['in_channel'],
            spk_num=spk_num,
            nonlinear=params['nonlinear'],
        )
    elif name == 'uttl':
        return UTT(
            params=torch.load(params['model_path'])
        )
    elif name == 'uttlex':
        return UTTEX(
            params=torch.load(params['model_path']),
            type=params['u3type'] if 'u3type' in params else '111',
            diag=params['diag'] if 'diag' in params else False
        )
    else:
        raise Exception('Model type {} hasn\'t defined in confParser/modelParser.'.format(name))
