# @Time : 2021/4/6 8:26
# @Author : Yangcheng Wu
# @FileName: basic_model.py
from dataDealer.collate_fns import *

collateMapper = {
    # note that default_collate performs different on different version of pytorch
    'default': default_collate,
    'invariant': invariant_collate,
    'invariant_align': invariant_align_collate,
    'min': min_align_collate,
    'fix_crop': param_collate.fix_crop,
}