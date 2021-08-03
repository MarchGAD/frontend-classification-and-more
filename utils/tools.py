# @Time : 2021/4/7 17:13
# @Author : Yangcheng Wu
# @FileName: tools.py
import os
import re
import torch
import random


def quadra(x1, A, x2):
    return ((x1 @ A).unsqueeze(1) @ x2.unsqueeze(2)).squeeze()


def ifelsedict(key, default, params):
    return params[key] if key in params else default


def epoch_control(dir_path, prefix, save_epochs):
    ls = os.listdir(dir_path)
    tmpls = []
    for name in ls:
        model_path = os.path.join(dir_path, name)
        # ignore directories under the 'dirpath'
        if os.path.isfile(model_path):
            find = re.match(prefix + '_epoch_([0-9]+)', name)
            if find is None:
                continue
            else:
                tmpls.append((name, int(find.group(1))))
    tmpls = sorted(tmpls, key=lambda x:x[1])
    sub = len(tmpls) - save_epochs
    if sub <= 0:
        pass
    else:
        for name in [i[0] for i in tmpls[:- save_epochs]]:
            os.remove(os.path.join(dir_path, name))


def seed_torch(seed=114514):
    '''
    https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

