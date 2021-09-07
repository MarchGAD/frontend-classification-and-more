# @Time : 2021/4/7 21:26
# @Author : Yangcheng Wu
# @FileName: main.py
import os
import torch
import shutil
import logging
import time
import torch.utils.data as Data
from argparse import ArgumentParser
from confParser.mainParser import MainParser
from pipeline.training import train, valid, test
from utils.tools import epoch_control
from utils.lr_schedule import LRS


def get_loaders(parser, args):
    params = parser.configs
    train_loader = Data.DataLoader(
        dataset=parser.trainset,
        batch_size=params['train']['batch_size'],
        collate_fn=parser.collatefn,
        shuffle=not args.multi_gpu,
        num_workers=5,
    )
    valid_loader = Data.DataLoader(
        dataset=parser.trainset,
        batch_size=params['valid']['batch_size'],
        collate_fn=parser.collatefn,
        shuffle=False,
        num_workers=5
    )
    test_loader = Data.DataLoader(
        dataset=parser.trainset,
        batch_size=params['test']['batch_size'],
        collate_fn=parser.collatefn,
        shuffle=False,
        num_workers=5
    )
    return train_loader, valid_loader, test_loader


def main():
    parser = ArgumentParser()
    parser.add_argument('config_path', type=str, default='../confs/tdnn.ini')
    parser.add_argument('-c', '--cards', type=int, nargs='+', default=[0])
    parser.add_argument('-ep', '--exp_path', type=str, default='exp')
    parser.add_argument('--multi-gpu', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cards).strip('[]')
    mainP = MainParser(args.config_path, exp_path=args.exp_path, main_path='.')
    params = mainP.configs
    model_path, recorder = mainP.model_path, mainP.recorder
    shutil.copyfile(__file__, os.path.join(model_path, 'train.py'))
    shutil.copyfile(args.config_path, os.path.join(model_path, 'config.ini'))
    lr = LRS()
    trainl, validl, testl = get_loaders(mainP, args)

    for i in range(params['train']['epoch']):
        train(dataloader=trainl,
              optimizer=mainP.optimizer,
              model=mainP.model,
              lossfn=mainP.loss.loss_func,
              epoch_num=i,
              recorder=recorder,
              show_gap=params['train']['show_gap'],
              restrict=mainP.restrict)
        torch.save(mainP.model, os.path.join(model_path, '{}_epoch_{}'.
                                             format(params['model']['type'], i)))
        if 'maintain_epochs' in params['exp']:
            epoch_control(model_path, params['model']['type'], params['exp']['maintain_epochs'])

        valacc, valloss = valid(
            dataloader=validl,
            model=mainP.model,
            lossfn=mainP.loss.loss_func
        )
        logging.info('epoch {}, valid_acc: {}, valid_loss: {}'.format(i, valacc, valloss))

        testacc, testloss = test(
            dataloader=testl,
            model=mainP.model,
            lossfn=mainP.loss.loss_func
        )
        logging.info('epoch {}, test_acc: {}, test_loss: {}'.format(i, testacc, testloss))
        lr.add(valacc, valloss, testacc, testloss)
        lr.half_scuedule(5, mainP.optimizer)



