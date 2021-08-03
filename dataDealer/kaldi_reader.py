# @Time : 2021/4/6 8:06
# @Author : Yangcheng Wu
# @FileName: kaldi_reader.py
from kaldiio import load_scp
import torch
import re
import logging


class KaldiReader:

    def __init__(self, scp_path, spkpattern=r'(.*?)-.*', group=1, pre_load=False):
        self.scp_path = scp_path

        self.pre_load = pre_load
        self.spk_pattern = spkpattern
        self.group = group
        self.scpX = self.scp_path[-4:] == 'scpx'
        self.datas = load_scp(self.scp_path)
        self.utts = sorted([_ for _ in self.datas.keys()])
        self.__generate_dics__()

    def __generate_dics__(self):
        self.spk2utt = {}
        self.utt2spk = {}
        for utt in self.utts:
            spk = re.match(self.spk_pattern, utt).group(self.group)
            # make sure there's no duplicate utts
            assert utt not in self.utt2spk
            self.utt2spk[utt] = spk
            if spk not in self.spk2utt:
                self.spk2utt[spk] = []
            self.spk2utt[spk].append(utt)
        self.spk_num = len(self.spk2utt)
        self.spk2label = {spk: i for i, spk in enumerate(sorted(self.spk2utt.keys()))}
        logging.info('{} has {} speakers and {} utterances.'.
                     format(self.scp_path, self.spk_num, len(self.utts)))
        if self.pre_load:
            self.hub = {utt: torch.from_numpy(self.datas.get(utt)).transpose(0, 1) for utt in self.utts}

    def get_xvec(self, item):
        utt = self.utts[item]
        if self.pre_load:
            xvec = self.hub[utt]
        else:
            xvec = torch.from_numpy(self.datas.get(utt))
            if len(xvec.size()) > 1:
                xvec = xvec.transpose(0, 1)
        return xvec

    def specified_spklabels(self, spk2label: dict):
        assert len(spk2label) == self.spk_num
        for spk in spk2label:
            assert spk in self.spk2label
        self.spk2label = spk2label


if __name__ == '__main__':
    path = '../scps/demo/test.scp'
    a = KaldiReader(path, pre_load=True)
    print(a.utts[0])
    c = a.get_xvec(0)
    print(c)