import torch
import torch.utils.data as Data


class TMP(Data.Dataset):

    def __init__(self, k):
        self.data = k

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def collate_fn(batch_data):
    # a = torch.stack(batch_data, 0)
    return batch_data, 1000


if __name__ == '__main__':
    from torch.utils.data.dataloader import default_collate
    a = [[1, 2, 3, 4, 5],
         [1, 2, 3],
         [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
         [1, 4, 23, 23, 2, 3],
         [1, 3, 4, 5, 2, 6, 7]]
    print(a)

    a = TMP(a)
    tl = Data.DataLoader(
        dataset=a,
        shuffle=True,
        batch_size=2,
        collate_fn=collate_fn
    )
    for (cnt, i) in enumerate(tl):
        print(cnt, i)

    # import torch.nn as nn
    #
    # inp = torch.rand(3, 10, 4)
    # a = torch.rand(1, 10, 20)
    # b = torch.rand(1, 10, 4)
    # c = torch.rand(1, 10, 5)
    # print(a)
    # print(b)
    # print(c)
    # m = nn.Conv1d(10, 1, 3)
    # ans = [m(a), m(b), m(c)]
    # print('*********************')
    # for i in ans:
    #     print(i)
    # print('++++++++++++++++++')
    # # t = torch.zeros(*b.size()[:-1], - b.size(-1))
    # # print(t.size())
    # # print(b.size())
    # b = torch.cat([b, torch.zeros(*b.size()[:-1], 20 - b.size(-1))], dim=-1)
    # c = torch.cat([c, torch.zeros(*c.size()[:-1], 20 - c.size(-1))], -1)
    # print(b.size())
    # print(c.size())
    # ans = m(torch.stack((a, b, c)).squeeze())
    # print(ans)