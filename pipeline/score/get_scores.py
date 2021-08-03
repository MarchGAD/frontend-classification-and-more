# @Time : 2021/4/28 15:23
# @Author : Yangcheng Wu
# @FileName: get_scores.py
import sys
import os
dirpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirpath + '/../../.')
from model.siaUTT import *


if __name__ == '__main__':
    from Scorer import Scorer
    from Resulter import Resulter
    import multiprocessing as mp

    scp_path = '../../scps/xmu/test.scp'
    model_path = '/data/wuyangcheng/Frontend-Classification/exp/utt-l/nnet/uttl_epoch_99'
    trials_path = '../pipeline/score/trials'
    scorer = Scorer(model_path, trials_path, scp_path, batch_size=128, use_gpu=True,
                    model=cosineUTTL(model_path), mgr=mp.Manager())
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