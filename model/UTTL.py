# @Time : 2021/4/17 23:27
# @Author : Yangcheng Wu
# @FileName: UTTL.py
import torch.nn as nn
import torch
from model.basic_model import BasicModel, nonlinearMapper


class UTT(BasicModel):

    def __init__(self, params):
        super(UTT, self).__init__()
        self.params = params
        self.bn6 = nn.BatchNorm1d(512)
        self.tdnn7 = nn.Linear(512, 512)
        self.softmax = nn.Linear(512, 7323)

        self._weight_init()

    def register(self, **named_parameters):
        for name, val in named_parameters.items():
            self.register_parameter(name, val)

    def _weight_init(self):
        self.bn6.running_mean.data = params['tdnn6.batchnorm.running_mean']
        self.bn6.running_var.data = params['tdnn6.batchnorm.running_var']
        self.bn6.track_running_stats = params['tdnn6.batchnorm.num_batches_tracked']

        self.tdnn7.weight.data = params['tdnn7.affine.weight'].squeeze()
        self.tdnn7.bias.data = params['tdnn7.affine.bias'].squeeze()
        self.softmax.weight.data = params['loss.affine.weight'].squeeze()
        self.softmax.bias.data = params['loss.affine.bias'].squeeze()

    def extract(self):
        pass

    def forward(self, x):
        x = self.bn6(x)
        x = self.tdnn7(x)
        x = self.softmax(x)
        return x





if __name__ == '__main__':
    import kaldiio as ko
    import re
    from loss.losses import Softmax
    params_path = '/data/wuyangcheng/kaldi-master/egs/xmuspeech/vox/exp/vox1_spec_am_reduceP_sgd/20.params'
    scp_path = '/data/wuyangcheng/kaldi-master/egs/xmuspeech/vox/exp/vox1_spec_am_reduceP_sgd/near_epoch_20_1st/train/xvector.scp'
    spk2label_path = './tmpspk2label'


    with open(spk2label_path, 'r') as f:
        spk2label = {}
        for line in f:
            spk, label = line.strip().split()
            spk2label[spk] = int(label)
    scpdic = ko.load_scp(scp_path)
    tmp = 0
    thres = 100
    input = None
    labels = []
    for k in scpdic:
        tmp += 1
        if tmp > thres:
            break
        xvec = torch.from_numpy(scpdic.get(k)).unsqueeze(0)
        label = spk2label[re.match(r'(.*?)-', k).group(1)]
        labels.append(label)
        if input is None:
            input = xvec
        else:
            input = torch.cat([input, xvec], 0)
    labels = torch.tensor(labels)
    params = torch.load(params_path)

    # model = UTT(params)
    modelpath = '/data/wuyangcheng/Frontend-Classification/exp/utt-l/uttl_epoch_68'
    model = torch.load(modelpath)
    ans = model(input)
    loss = Softmax().loss_func(ans, labels)
    print(loss)
    print(torch.sum(labels == torch.argmax(ans)).float() / labels.size(0))



