"""
author:   Khanh Lam
ref:      https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras
"""

import numpy as np
import cv2
import os
import fnmatch
import keras
import keras.backend as K
import argparse

from keras import Model

from keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    Lambda,
    Bidirectional,
    LSTM,
    Dense,
)


from utils import CHAR_LIST, resize_img


def reload_model():
    # input with shape of height=32 and width=128
    inputs = Input(shape=(32, 128, 1))

    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    # pooling layer with kernel size (2,2)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool_2)

    conv_4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv_3)
    # pooling layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool_4)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3, 3), activation="relu", padding="same")(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

    conv_7 = Conv2D(512, (2, 2), activation="relu")(pool_6)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

    outputs = Dense(len(CHAR_LIST) + 1, activation="softmax")(blstm_2)

    act_model = Model(inputs, outputs)
    act_model.load_weights("best_model.hdf5")

    return act_model


def main(args):
    act_model = reload_model()

    test_img = []
    test_label = []

    for root, dirnames, list_files in os.walk(args["image_dir"]):
        for file_name in list_files:
            img = cv2.cvtColor(
                cv2.imread(os.path.join(root, file_name)), cv2.COLOR_BGR2GRAY
            )
            img = resize_img(img)

            txt = file_name.split("_")[1]

            test_label.append(txt)
            test_img.append(img)

    test_img = np.array(test_img)
    # predict outputs on validation images
    prediction = act_model.predict(test_img)

    # use CTC decoder
    out = K.get_value(
        K.ctc_decode(
            prediction,
            input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
            greedy=True,
        )[0][0]
    )

    print("\nResult---------------------\n")
    # see the results
    i = 0
    for x in out:
        print("original text  =", test_label[i])
        print("predicted text = ", end="")
        for p in x:
            if int(p) != -1:
                print(CHAR_LIST[int(p)], end="")
        print("\n")
        i += 1


ap = argparse.ArgumentParser()
ap.add_argument(
    "-dir",
    "--image-dir",
    type=str,
    required=True,
    help="Path to input image to predict",
)
args = vars(ap.parse_args())

if __name__ == "__main__":
    main(args)