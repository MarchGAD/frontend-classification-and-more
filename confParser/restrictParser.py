# @Time : 2021/5/2 9:58
# @Author : Yangcheng Wu
# @FileName: restrictParser.py
from restrict.restricts import *

restrictMapper = {
    'semi': semior,
    'ffsemi': ffsemior,
    'diag': diag
}

def getRestrictFunc(name):
    return restrictMapper[name]