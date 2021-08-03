# @Time : 2021/4/7 17:33
# @Author : Yangcheng Wu
# @FileName: training.py
from argparse import ArgumentParser
import torch
import logging


def calculate_acc(label, ans):
    return torch.sum(label == torch.argmax(ans, dim=-1)).float() / label.size(0)


def train(dataloader, optimizer, model, lossfn, epoch_num, show_gap, recorder, restrict=None):
    model.train()
    for step, (data, label) in enumerate(dataloader):
        data = data.cuda()
        label = label.cuda()
        losses = []
        optimizer.zero_grad()
        ans = model(data)
        acc = calculate_acc(label, ans)
        loss = lossfn(ans, label)
        losses.append(('normal', loss.item()))
        for name, params in model.named_parameters():
            for restrict_name in restrict:
                if name in restrict[restrict_name]['targets']:
                    tmp_loss = restrict[restrict_name]['func'](params)
                    loss += tmp_loss
                    losses.append((restrict_name, name, tmp_loss.item()))
        if not step % show_gap:
            recorder.add_scalar('training_loss', loss.item(),
                                global_step=epoch_num * (dataloader.dataset.__len__() // dataloader.batch_size)
                                + step)
            info = 'epoch:{}, step:{}, loss:{}, acc:{}%'.format(epoch_num, step, losses[0][1], 100 * acc)
            if len(losses) > 1:
                for i in losses[1:]:
                    info += ', {} on {}:{}'.format(*i)
            logging.info(info)

        if loss > 0:
            loss.backward()
            optimizer.step()


def valid(dataloader, model, lossfn):
    model.eval()
    tot = 0.0
    mean_loss = 0.0
    tot_num = 0.0
    for step, (data, label) in enumerate(dataloader):
        data = data.cuda()
        label = label.cuda()
        ans = model(data)
        acc = calculate_acc(label, ans)
        tot += acc.item() * data.size(0)
        loss = lossfn(ans, label)
        mean_loss += loss.item() * data.size(0)
        tot_num += data.size(0)
    return tot / tot_num, mean_loss / tot_num


def test(dataloader, model, lossfn):
    model.eval()
    tot = 0.0
    mean_loss = 0.0
    tot_num = 0.0
    for step, (data, label) in enumerate(dataloader):
        data = data.cuda()
        label = label.cuda()
        ans = model(data)
        acc = calculate_acc(label, ans)
        tot += acc.item() * data.size(0)
        loss = lossfn(ans, label)
        mean_loss += loss.item() * data.size(0)
        tot_num += data.size(0)
    return tot / tot_num, mean_loss / tot_num

