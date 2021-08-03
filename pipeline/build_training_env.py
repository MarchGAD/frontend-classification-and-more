# @Time : 2021/4/8 13:25
# @Author : Yangcheng Wu
# @FileName: build_training_env.py
import os
import shutil
import logging
from argparse import ArgumentParser
from tensorboardX import SummaryWriter


def build(exp_path, proj_name, main_path):
    proj_path = os.path.join(main_path, exp_path, proj_name)
    model_path = create_exp(proj_path, main_path)
    recorder = loss_record(proj_path)
    return model_path, recorder


def create_exp(proj_path, main_path):
    code_dirs = ['confParser', 'dataDealer', 'loss', 'model', 'pipeline', 'utils']
    os.mkdir(proj_path)
    model_path = os.path.join(proj_path, 'nnet')
    os.mkdir(model_path)
    create_log(os.path.join(proj_path, 'log'))
    for dir in code_dirs:
        shutil.copytree(os.path.join(main_path, dir), os.path.join(proj_path, dir))
    return model_path


def loss_record(path):
    recorder = SummaryWriter(path)
    return recorder


def create_log(log_path):
    logging.basicConfig(filename=log_path, level=0,
                        format='%(asctime)s-%(message)s',
                        datefmt='%H:%M:%S')


def main():
    parser = ArgumentParser()
    parser.add_argument('exp_path')
    parser.add_argument('proj_name')
    parser.add_argument('root')
    return parser.parse_args()

if __name__ == '__main__':
    args = main()
    build(args.exp_path, args.proj_name, args.root)