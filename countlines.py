import os


def cntline(file):
    tcnt = 0
    with open(file, 'rb') as f:
        for _ in f:
            tcnt += 1
    return tcnt


ign = ['scps', 'voxceleb_id10001', 'exp']
sta = ['.']

cnt = 0
while len(sta) > 0:
    pat = sta.pop()
    ts = os.listdir(pat)
    for t in ts:
        tmppat = os.path.join(pat, t)
        if os.path.isdir(t):
            sta.append(tmppat)
        elif t[-3:] == '.py':
            print(t)
            cnt += cntline(tmppat)
print(cnt)