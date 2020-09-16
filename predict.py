import numpy as np

from train_utils import load_video_file_data
import tensorflow as tf
import argparse


def predict_one(predict_model, data_file):
    vid_mid = load_video_file_data(data_file)
    vid_mid_list = [vid_mid]
    vid_mid_list = np.array(vid_mid_list)
    print("vid_mid_list--------->")
    print(vid_mid_list.shape)
    y = predict_model.predict(vid_mid_list)
    return y


def main(input_data_path):
    new_model = tf.keras.models.load_model('model.h5')
    y = predict_one(new_model, input_data_path)
    print(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Sign language with Mediapipe')
    parser.add_argument("--input_data_path",help=" ")
    args=parser.parse_args()
    input_data_path=args.input_data_path
    main(input_data_path)