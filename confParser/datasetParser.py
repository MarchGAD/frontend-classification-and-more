# @Time : 2021/4/7 17:10
# @Author : Yangcheng Wu
# @FileName: datasetParser.py
from dataDealer.kaldi_dataset import KaldiSet
from utils.tools import ifelsedict


def getdataset(params, scp_key=None):
    if params['datasettype'] == 'kaldiset':
        return KaldiSet(
            scp_path=params['scp_path'] if scp_key is None else params[scp_key],
            pre_load=ifelsedict('pre_load', False, params),
            spkpattern=ifelsedict('spkpattern', r'(.*?)-.*', params),
            group=ifelsedict('group', 1, params),
            strategy=ifelsedict('strategy', 'random', params)
        )