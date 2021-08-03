# @Time : 2021/4/6 9:16
# @Author : Yangcheng Wu
# @FileName: mainParser.py
from configparser import ConfigParser
import os
import re
import time
import logging
import torch
import torch.utils.data as Data
from collections import OrderedDict as od
from confParser.lossParser import getloss
from confParser.modelParser import getmodel
from confParser.datasetParser import getdataset
from confParser.optParser import getoptimizer
from confParser.restrictParser import getRestrictFunc
from dataDealer.basic_datadealer import collateMapper
from pipeline.build_training_env import build
from utils.tools import seed_torch

pattern_mapper = od(
    {
        'string': r'\'(.*)\'',
        'true': r'^[t|T]rue$',
        'false': r'^[fin|F]alse$',
        'int': r'^[-|+]?[0-9]+$',
        'float': r'^[-|+]?[0-9]+\.[0-9]*$',
        'scientific notation': r'^[-|+]?[0-9]+(\.[0-9]*)?e[-|+]?[0-9]+$',
        'list': r'^\[?(((.*?),)+(.*?)*)\]?$'
    }
)
transform_mapper = {
    'int': int,
    'float': float,
    'scientific notation': float,
}
project_mapper = {
    'true': True,
    'false': False
}


class MainParser:

    def __init__(self, conf_path, exp_path, main_path):
        parser = ConfigParser()
        parser.read(conf_path)
        self.configs = {sec:{opt: parser[sec][opt] for opt in parser.options(sec)} for sec in parser.sections()}
        for sec in self.configs:
            for opt in self.configs[sec]:
                self.configs[sec][opt] = self.parse(self.configs[sec][opt])
        # logging should be used after being built
        self.model_path, self.recorder = build(exp_path, self.configs['exp']['name'], main_path)

        # self.seed = time.time()

        self.seed = 19990110
        seed_torch(self.seed)
        logging.info('seed is {}'.format(self.seed))

        self.trainset = getdataset(self.configs['train'])
        self.validset = getdataset(self.configs['valid'])
        self.testset = getdataset(self.configs['test'])

        self.loss = getloss(self.configs['loss'])
        self.model = getmodel(self.configs['model'], spk_num=self.trainset.reader.spk_num)
        self.restrict = {}
        for name in self.configs['restrict']:
            self.restrict[name] = {}
            self.restrict[name]['targets'] = self.configs['restrict'][name]
            self.restrict[name]['func'] = getRestrictFunc(name)

        self.register_loss_into_model()

        if self.configs['train']['use_gpu']:
            self.model.cuda()
        logging.info(self.model)

        self.optimizer = getoptimizer(self.configs['optimizer'], self.model)

        collate_key = self.configs['exp']['collate'] if 'collate' in self.configs['exp'] else 'default'
        self.collatefn = collateMapper[collate_key]

    def register_loss_into_model(self):
        self.model.register(**self.loss.lparams)

    @staticmethod
    def parse(string):
        ans = string

        for type, pattern in pattern_mapper.items():
            tmp = re.match(pattern, string)
            if tmp is not None:
                if type == 'string':
                    ans = tmp.group(1)
                elif type == 'list':
                    ans = [MainParser.parse(i.strip()) for i in tmp.group(1).split(',')]
                elif type in project_mapper:
                    ans = project_mapper[type]
                elif type in transform_mapper:
                    ans = transform_mapper[type](string)
                else:
                    raise Exception('Please complete the relation in type: {}'.format(type))
                break
        return ans

    def multi_gpu(self):
        # import horovod.torch as hvd
        # hvd.init()
        # # Horovod: pin GPU to local rank.
        # torch.cuda.set_device(hvd.local_rank())
        # # Partition dataset among workers using DistributedSampler
        # logging.info('Using multi-gpu for training.')
        # train_sampler = Data.distributed.DistributedSampler(
        #     self.trainset, num_replicas=hvd.size(), rank=hvd.rank())
        #
        # # Horovod: scale learning rate by the number of GPUs.
        # self.configs['optimizer']['lr'] *= hvd.size()
        #
        # # Horovod: wrap optimizer with DistributedOptimizer.
        # self.optimizer = hvd.DistributedOptimizer(
        #     getoptimizer(self.configs['optimizer'], self.model)
        # , self.model.named_parameters()
        # )
        #
        # # Horovod: broadcast parameters & optimizer state.
        # hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        # hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        #
        # return train_sampler
        torch.distributed.init_process_group('nccl', init_method='env://')
        torch.cuda.set_device(0)
        device = torch.device('cuda', 0)
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
        # self.model.to()

        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                   device_ids=list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))),
                                   output_device=0)
        return train_sampler, device



if __name__ == '__main__':
    a = MainParser('../confs/uttlex.ini', None, None)
    print(a.configs)

    class A:
        def __init__(self, **kwargs):
            pass
    a = A(**{'1':2, '3':4})