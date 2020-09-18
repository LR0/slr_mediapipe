import numpy as np

from train_utils import load_predict_file_data
import tensorflow as tf
import argparse


# prediction
def load_label():
    listfile = []
    with open("label.txt", mode='r') as l:
        listfile = [i for i in l.read().split()]
    label = {}
    count = 1
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label


def predict_one(predict_model, data_file):
    vid_mid = load_predict_file_data(data_file)
    vid_mid_list = [vid_mid]
    vid_mid_list = np.array(vid_mid_list)
    print("vid_mid_list--------->")
    print(vid_mid_list.shape)
    y = predict_model.predict(vid_mid_list)
    return y


def main(input_data_path):
    new_model = tf.keras.models.load_model('model.h5')
    y = predict_one(new_model, input_data_path)
    labels = load_label()
    print(y)
    predictions = np.array([np.argmax(pred) for pred in y])
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    for i in predictions:
        print(rev_labels[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Sign language with Mediapipe')
    parser.add_argument("--input_data_path", help=" ")
    args = parser.parse_args()
    input_data_path = args.input_data_path
    main(input_data_path)
