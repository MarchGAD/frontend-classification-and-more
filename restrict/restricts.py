# @Time : 2021/5/2 9:59
# @Author : Yangcheng Wu
# @FileName: restricts.py
import torch


def ffsemior(mat, use_gpu=True):
    P = mat @ mat.transpose(0, 1)
    PPT = P @ P.transpose(0, 1)
    alpha = torch.trace(PPT) / torch.trace(P)
    P = 1.0 / alpha * mat @ mat.transpose(0, 1)
    I = torch.diag(torch.ones(mat.size(0)))
    if use_gpu:
        I = I.cuda()
    Q = P - I
    return torch.trace(Q @ Q.transpose(0, 1))


def semior(mat, use_gpu=True):
    P = mat @ mat.transpose(0, 1)
    I = torch.diag(torch.ones(mat.size(0)))
    if use_gpu:
        I = I.cuda()
    Q = P - I
    return torch.trace(Q @ Q.transpose(0, 1))


def diag(mat):
    assert mat.size(0) == mat.size(1)
    A = mat - mat.transpose(0, 1)
    ATA = A.transpose(0, 1) @ A
    trace = torch.trace(ATA)
    return trace

if __name__ == '__main__':

    '''
        test for diag
    '''
    torch.manual_seed(19990110)
    inp1 = torch.rand(1, 5)
    inp2 = torch.rand(1, 5)
    mid = torch.rand(5, 5).requires_grad_(True)

    print(mid)
    print('*****************')
    for i in range(100):
        grad = torch.autograd.grad(diag(mid), mid)
        mid = mid - 0.01 * grad[0].detach()
        print(mid)
        trace = diag(mid)
        print(trace)
        print('******************')

