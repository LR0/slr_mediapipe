from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models
from keras.models import Sequential
from keras import layers
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import random
from keras import optimizers
from keras.layers import SimpleRNN, Dense
from keras.layers import Bidirectional
import os
import sys
from keras.layers import Masking, Embedding


def make_label(text):
    with open("label.txt", "w") as f:
        f.write(text)
    f.close()


# 最后返回的应该是一个(15*8, 21*2*3)的np矩阵
# 所有文件的输入都会被规范化为如此的大小，超过切割，不足填充0
def load_video_file_data(file_path):
    video_hands_mid = []
    # 一个视频一个文件
    with open(file_path, mode='r') as t:
        # 一行一帧
        for line in t:
            numbers = [float(num) for num in line.split()]
            if len(numbers) < 1:
                continue
            # 最多三只手，不足用-1填充
            for i in range(len(numbers), 126):
                numbers.extend([0.000])
            video_hands_mid.append(numbers[0:126])

    gap_arr = np.zeros(126)
    # 将帧数补足到120帧，样本均为15帧
    for i in range(len(video_hands_mid), 15*8):
        video_hands_mid.append(gap_arr)

    video_hands_mid = np.array(video_hands_mid[0:15*8])
    print("video_hands_mid.shape-->")
    print(video_hands_mid.shape)
    return video_hands_mid


def load_label_data(dir_path):
    print("load_label_data------------->")
    print(dir_path)
    if dir_path[-1] != '/':
        dir_path = dir_path + '/'
    file_list = os.listdir(dir_path)

    # 一个矩阵代表一个视频材料
    label_video_mid_list = []
    # 文件夹名为label名
    for file_name in file_list:
        print("file_name------------>")
        print(file_name)
        # 一个文本文件一个视频
        if os.path.splitext(file_name)[-1] != ".txt":
            continue

        file_path = dir_path+file_name
        print("file_path------------>")
        print(file_path)
        video_mid = load_video_file_data(file_path)
        video_mid = np.array(video_mid)
        print("vid_mid_np----------------->")
        print(video_mid.shape)
        label_video_mid_list.append(video_mid)

    # 返回三维矩阵列表，每个矩阵的形状必须一摸一样
    return label_video_mid_list


def load_data(dirname):
    if dirname[-1] != '/':
        dirname = dirname + '/'
    listfile = os.listdir(dirname)
    X = []
    Y = []
    XT = []
    YT = []

    # 一个文件夹一个标签
    for label_name in listfile:
        # 不是文件夹则跳过
        dir_path = dirname + label_name
        if not os.path.isdir(dir_path):
            continue

        label_vid_mid_list = load_label_data(dir_path)
        print("label_vid_mid_list.len------>")
        print(len(label_vid_mid_list))

        # 遍历视频信息矩阵
        for i in range(len(label_vid_mid_list)):
            print(i)
            # 三分之一的测试集
            if i % 3 == 1:
                # 一个测试视频矩阵对应一个标签
                XT.append(label_vid_mid_list[i])
                YT.append(label_name)
                continue

            # 一个训练视频矩阵对应一个标签
            X.append(label_vid_mid_list[i])
            Y.append(label_name)

    X = np.array(X)
    print("X_shape--------------->")
    print(X.shape)
    Y = np.array(Y)
    print("Y_shape--------------->")
    print(Y.shape)
    XT = np.array(XT)
    YT = np.array(YT)

    # 乱序
    tmp = [[x, y] for x, y in zip(X, Y)]
    random.shuffle(tmp)

    tmp1 = [[xt, yt] for xt, yt in zip(XT, YT)]
    random.shuffle(tmp1)

    X = [n[0] for n in tmp]
    Y = [n[1] for n in tmp]
    XT = [n[0] for n in tmp1]
    YT = [n[1] for n in tmp1]

    k = set(Y)
    ks = sorted(k)
    text=""
    for i in ks:
        text=text+i+" "
    make_label(text)

    s = Tokenizer()
    s.fit_on_texts([text])
    encoded = s.texts_to_sequences([Y])[0]
    encoded1 = s.texts_to_sequences([YT])[0]
    one_hot = to_categorical(encoded)
    one_hot2 = to_categorical(encoded1)

    (x_train, y_train) = X, one_hot
    (x_test, y_test) = XT, one_hot2
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print("X_tmp_shape--------------->")
    print(x_train.shape)
    print("Y_tmp_shape--------------->")
    print(y_train.shape)
    return x_train, y_train, x_test, y_test


def build_model(label):
    model = Sequential()

    # 补全帧数，默认10秒15帧
    # model.add(Masking(mask_value=-1, input_shape=(15*10, 21*2*3,)))
    # timestep为10秒，一帧最多三只手21*2*3=126维
    model.add(layers.LSTM(64, return_sequences=True,
                          input_shape=(15*8, 21*2*3)))
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.LSTM(32))

    # 根据label数确定全连接层
    model.add(layers.Dense(label, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
