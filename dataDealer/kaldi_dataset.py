# @Time : 2021/4/6 8:22
# @Author : Yangcheng Wu
# @FileName: kaldi_dataset.py
from dataDealer.kaldi_reader import KaldiReader
import torch.utils.data as Data
import logging
import re

note = {
    'random': 'totally random, without concering labels(spks)'
}


class KaldiSet(Data.Dataset):

    def __init__(self, scp_path, pre_load=False, spkpattern=r'(.*?)-.*', group=1,
                 strategy='random'):
        super(KaldiSet, self).__init__()
        logging.info('Choose {} strategy: {}'.format(strategy, note[strategy]))
        self.reader = KaldiReader(scp_path=scp_path,
                                  spkpattern=spkpattern,
                                  group=group,
                                  pre_load=pre_load)
        self.strategy = strategy
        self.uttshapes = []
        if self.reader.scpX:
            for utt in self.reader.utts:
                shapes = re.match(r'.*?_shape_(.*)', utt).group(1)
                tmp_shape = [int(i) for i in shapes.split('_')]
                self.uttshapes.append(tuple(tmp_shape))

        self.len_mapper = {
            'random': len(self.reader.utts),

        }

        self.get_mapper = {
            'random': self.random_getter,

        }

    def random_getter(self, item):
        utt = self.reader.utts[item]
        spk = self.reader.utt2spk[utt]
        return self.reader.get_xvec(item), self.reader.spk2label[spk]

    def trim_short_utt(self, length):
        assert self.reader.scpX
        death_list = []
        logging.info('Trimming utts less than {}.'.format(length))
        for i, shape in enumerate(self.uttshapes):
            if shape[0] < length:
                death_list.append(i)
        for ind in death_list.__reversed__():
            utt = self.reader.utts.pop(ind)
            shape = self.uttshapes.pop(ind)
        self.reader.__generate_dics__()


    def __len__(self):
        # print(self.len_mapper[self.strategy])
        return self.len_mapper[self.strategy]

    def __getitem__(self, item):
        return self.get_mapper[self.strategy](item)


if __name__ == '__main__':
    import torch.utils.data as Data
    # train_loader = Data.DataLoader(
    #     dataset=KaldiSet(
    #         '../scps/demo/train.scp'
    #     ),
    #     batch_size=100,
    #     shuffle=True,
    # )
    #
    # for i, data in enumerate(train_loader):
    #     print(i, data)
    a = KaldiSet(
        '../scps/demo/test.scpx'
    )
    print(len(a.reader.utts))
    print(len(a.uttshapes))
    a.trim_short_utt(600)
    print(len(a.reader.utts))
    print(len(a.uttshapes))
