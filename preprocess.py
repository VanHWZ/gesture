import wave
import os
import numpy as np

'''
预处理，返回一个文件中[[amplitude1,amplitude2,..]]的np数组
'''
def process_single_file(name):
    f = wave.open(name, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    data = waveData[0]
    f.close()

    Fs = 44100  # sampling rate采样率
    y = waveData[0]

    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq1 = frq[range(int(n / 2))]  # one side frequency range

    Y = np.fft.fft(y) / n  # fft computing and normalization 归一化
    Y1 = Y[range(int(n / 2))]
    abs_Y1 = abs(Y1)

    l = []
    result = []
    result_max = [0, 0.0] # 用于取振幅最大的点

    for i in range(0, len(frq1)):
        l.append([frq1[i], abs_Y1[i]])
        if l[i][0] > 20750 and l[i][0] < 21250:
            result.append(l[i])
            if result_max[1] < l[i][1]:
                result_max = l[i]

    num = 60 # 一边的点数，总点数为2*num+1
    result = result[result.index(result_max)-num:result.index(result_max)+num+1]
    result_np = np.array([result])
    # print(result_np.T[1].T)
    return result_np.T[1].T

'''
处理一个目录中多个标签，返回X(样本数，样本长度), y(标签列表)
'''
def process_dir(dir):
    X = np.empty(shape=(0,121))
    list_y = []
    for i in os.listdir(dir):
        if os.path.isdir(dir+'/'+i):
            temp_dir = dir+'/'+i
            # print(len(os.listdir(temp_dir)))
            for files in os.listdir(temp_dir):
                X = np.concatenate((X, process_single_file(temp_dir+'/'+files)), axis=0)
                list_y.append(i)
    print(X.shape)
    y = np.array(list_y, dtype=str)
    print(y.shape)
    return X, y


if __name__ == "__main__":
    # process_single_file("./test/Awaytest001.wav")
    process_dir("./data")