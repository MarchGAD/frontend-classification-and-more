# @Time : 2021/4/8 16:33
# @Author : Yangcheng Wu
# @FileName: test.py

class Test:

    def __init__(self):
        pass

    def score(self):
        pass


if __name__ == '__main__':
    a = Test()
    if getattr(a, 'scor'):
        print('ha!')
    else:
        print('rua')