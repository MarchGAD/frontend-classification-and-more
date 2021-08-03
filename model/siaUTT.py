import sys
import os
dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirpath + '/../.')

import torch
import torch.nn as nn
from utils.tools import quadra
from model.UTTL import UTT
from model.UTTEX import UTTEX



class cosineUTTL(nn.Module):
    '''
        return cosine score from UTT
    '''

    def __init__(self, utt_path):
        super(cosineUTTL, self).__init__()
        self.utt = torch.load(utt_path)

    def pre_forward(self, x):
        x = self.utt.bn6(x)
        x = self.utt.tdnn7(x)
        return x

    def forward(self, *input):
        assert len(input) == 2
        x1 = self.pre_forward(input[0])
        x2 = self.pre_forward(input[1])
        xs = x1.unsqueeze(1) @ x2.unsqueeze(2)
        xs.squeeze_()
        return xs / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))


class cosineUTTEX(nn.Module):
    '''
        return cosine score from UTTEX
    '''

    def __init__(self, uttex_path):
        super(cosineUTTEX, self).__init__()
        self.uttex = torch.load(uttex_path)

    def pre_forward(self, x):
        x = self.uttex.bn6(x)
        x = self.uttex.tdnn7(x)
        return x

    def forward(self, *input):
        assert len(input) == 2
        x1 = self.pre_forward(input[0])
        x2 = self.pre_forward(input[1])
        xs = x1.unsqueeze(1) @ x2.unsqueeze(2)
        xs.squeeze_()
        return xs / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))


class LLRUTTEX(nn.Module):
    '''
        return LLR score from UTTEX
    '''

    def __init__(self, uttex_path):
        super(LLRUTTEX, self).__init__()
        self.utt = torch.load(uttex_path)

    def pre_forward(self, x):
        x = self.utt.bn6(x)
        x = self.utt.tdnn7(x)
        return x

    def forward(self, *input):
        assert len(input) == 2
        x1 = self.pre_forward(input[0])
        x2 = self.pre_forward(input[1])
        P = self.uttex.quadra_softmax.within
        Q = self.uttex.quadra_softmax.between
        c = self.uttex.quadra_softmax.other
        within = quadra(x1, P, x1) + quadra(x2, P, x2)
        bet = 2 * quadra(x1, Q, x2)
        oned = (x1 + x2) @ c
        oned.squeeze_()
        return within + bet + oned


if __name__ == '__main__':
    sys.path.append('../pipeline/.')
    from pipeline.score.Scorer import Scorer
    from pipeline.score.Resulter import Resulter

    scp_path = '../scps/xmu/test.scp'
    model_path = '/data/wuyangcheng/Frontend-Classification/exp/utt-l/nnet/uttl_epoch_99'
    trials_path = '../pipeline/score/trials'
    scorer = Scorer(model_path, trials_path, scp_path, batch_size=128, use_gpu=True,
                    model=cosineUTTL(model_path))
    scorer.get_all_scores()
    scorer.save_scores(dirpath + '/tmp_scores')
    basename = os.path.basename(os.path.dirname(os.path.dirname(dirpath)))
    fr = open('./result', 'a')
    print('***********************************', file=fr)
    print('tot is %d, miss %d, %f' % (scorer.tot, scorer.miss, scorer.miss / scorer.tot), file=fr)
    print('model_path is {}'.format(model_path), file=fr)

    a = Resulter(dirpath + '/tmp_scores',
                 trial_file=None,
                 trials=scorer.trials)
    print(a.compute_score(), file=fr)
    fr.close()