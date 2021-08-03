# @Time : 2021/4/15 12:24
# @Author : Yangcheng Wu
# @FileName: scp2scpx.py
import os
from argparse import ArgumentParser
from tqdm import tqdm
from kaldiio import load_scp


def main():
    parser = ArgumentParser()
    parser.add_argument('scp_path', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    dirpath = os.path.dirname(args.scp_path)
    name = os.path.basename(args.scp_path)
    pre_dic = load_scp(args.scp_path)
    aft_dic = {}
    firstline = True
    with open(args.scp_path, 'r') as fin:
        with open(args.scp_path + 'x', 'w') as fout:
            for line in tqdm(fin):
                key, path = line.strip().split()
                shape = pre_dic.get(key).shape
                key += '_shape'
                for j in shape:
                    key += '_{}'.format(j)
                fout.write('{}{} {}'.format('' if firstline else '\n', key, path))
                if firstline:
                    firstline = False
