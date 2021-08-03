# @Time : 2021/4/28 14:44
# @Author : Yangcheng Wu
# @FileName: kaldi_tester.py
import torch
from kaldiio import load_scp
import torch.utils.data as Data


class KaldiTester(Data.Dataset):

    def __init__(self, scp_path, trials, use_gpu=False, pre_load=False):
        super(KaldiTester, self).__init__()
        self.datas = load_scp(scp_path)
        self.tot = len(self.datas)
        self.use_gpu = use_gpu
        self.trials = trials
        self.miss = 0
        self.pre_load = pre_load
        if self.pre_load:
            self.data = {}
            for key in self.datas:
                self.data[key] = torch.from_numpy(self.datas.get(key))
                if self.use_gpu:
                    self.data[key] = self.data[key].cuda()

    def __getitem__(self, item):
        utt1, utt2, is_target = self.trials[item]
        if not self.pre_load:
            xvec1 = torch.from_numpy(self.datas.get(utt1))
            xvec2 = torch.from_numpy(self.datas.get(utt2))
            if self.use_gpu:
                xvec1 = xvec1.cuda()
                xvec2 = xvec2.cuda()
        else:
            xvec1 = self.data[utt1]
            xvec2 = self.data[utt2]
        return xvec1, xvec2, utt1, utt2

    def clear(self):
        i = -1
        while i < len(self.trials) - 1:
            i += 1
            utt1, utt2, is_target = self.trials[i]
            if utt1 in self.datas and utt2 in self.datas:
                continue
            else:
                self.trials.pop(i)
                self.miss += 1
                i -= 1

    def __len__(self):
        return len(self.trials)