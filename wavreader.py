import wave
import matplotlib.pyplot as plt

if __name__ == '__main__':
    t = wave.open('voxceleb_id10001/1zcIwhmdeo4/00001.wav')
    print(t.getnchannels(), t.getnframes(), t.getsampwidth(), t.getframerate())
    a = t.readframes(t.getnframes())
    import numpy as np
    a = np.fromstring(a, dtype=np.short)

    # time = np.arange(0, t.getnframes()) * (1 / t.getframerate())
    # plt.plot(time, a)
    plt.plot(a)
    plt.show()


    # print(t)
    # c = '\xfe\x08\xff\x08'.replace('\\', '0')
    # print(c)
