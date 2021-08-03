
tmp = ['100', '010', '001', '110', '101', '011']

for i in tmp:
    a = '''[exp]
name = uttlex-{}
collate = default

[optimizer]
type= sgd
lr = 4e-2
weight_decay = 1e-5

[model]
type = uttlex
in_channel = 512
model_path = /data/wuyangcheng/kaldi-master/egs/xmuspeech/vox/exp/vox1_spec_am_reduceP_sgd/20.params
u3type = {}

[loss]
type = softmax

[train]
scp_path =/data/wuyangcheng/Frontend-Classification/scps/xmu/train.scp
datasettype = kaldiset
use_gpu = false
batch_size = 128
epoch = 100
show_gap = 10
strategy = random

[valid]
scp_path =/data/wuyangcheng/Frontend-Classification/scps/xmu/validate.scp
datasettype = kaldiset
use_gpu = true
batch_size = 128

[test]
scp_path =/data/wuyangcheng/Frontend-Classification/scps/xmu/test.scp
datasettype = kaldiset
use_gpu = true
batch_size = 128

[restrict]
'''.format(i, i)
    with open('uttlex{}.ini'.format(i), 'w') as f:
        f.write(a)
