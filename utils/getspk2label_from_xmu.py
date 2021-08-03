# @Time : 2021/4/18 12:36
# @Author : Yangcheng Wu
# @FileName: getspk2label_from_xmu.py
import re
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xmu_egs', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    spk2label = {}
    with open(args.xmu_egs, 'r') as f:
        firstline = True
        for line in f:
            # ignore the first line of titles
            if firstline:
                firstline = False
                continue
            utt, path, s, e, label = line.strip().split()
            spk = re.match('(.*?)-', utt).group(1)
            label = int(label)
            if spk not in spk2label:
                spk2label[spk] = label
            else:
                if label != spk2label[spk]:
                    raise Exception('Multi-label for one single speaker detected.')
    with open('tmpspk2label', 'w') as f:
        for spk, label in spk2label.items():
            f.write('{} {}\n'.format(spk, label))
