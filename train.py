"""
author:     Khanh Lam
dataset:    https://www.robots.ox.ac.uk/~vgg/data/text/
ref:        https://github.com/TheAILearner/A-CRNN-model-for-Text-Recognition-in-Keras
"""

import os
import numpy as np
import fnmatch

import keras
import keras.backend as K

from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

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

from utils import CHAR_LIST, encode_to_labels, read_img


PATH_TO_DATASET = "C:/Users/quock/Downloads/Compressed/mnt/ramdisk/max/90kDICT32px"
NO_OF_IMAGES = 150000
MODEL_CK_POINT_FPATH = "best_model.hdf5"
BATCH_SIZE = 256
EPOCHS = 10

# lists for training dataset
training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

# lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

max_label_len = 0


# region Preprocess the data

i = 1  # var to count image loaded
flag = 0

print("\nStart preprocessing data ...")

for root, dirnames, list_files in os.walk(PATH_TO_DATASET):
    for file_name in fnmatch.filter(list_files, "*.jpg"):
        img = read_img(os.path.join(root, file_name))
        if img is None:
            continue

        txt = file_name.split("_")[1]

        if len(txt) > max_label_len:
            max_label_len = len(txt)

        # split data 10% for validation and 90% for training
        if i % 10 == 0:
            valid_orig_txt.append(txt)
            valid_label_length.append(len(txt))
            valid_input_length.append(31)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(txt))
        else:
            orig_txt.append(txt)
            train_label_length.append(len(txt))
            train_input_length.append(31)
            training_img.append(img)
            training_txt.append(encode_to_labels(txt))

        print(f"Loaded {i}/{NO_OF_IMAGES} image", end="\r")
        if i == NO_OF_IMAGES:
            flag = 1
            break
        i += 1
    if flag == 1:
        break

print("Preprocessing data done\n")

# endregion


# region Initial model

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

labels = Input(name="the_labels", shape=[max_label_len], dtype="float32")
input_length = Input(name="input_length", shape=[1], dtype="int64")
label_length = Input(name="label_length", shape=[1], dtype="int64")


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
    [outputs, labels, input_length, label_length]
)

# model to be used at training time
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
model.summary()

# endregion

model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer="adam")

# pad each output label to maximum text length
train_padded_txt = pad_sequences(
    training_txt, maxlen=max_label_len, padding="post", value=len(CHAR_LIST)
)
valid_padded_txt = pad_sequences(
    valid_txt, maxlen=max_label_len, padding="post", value=len(CHAR_LIST)
)

checkpoint = ModelCheckpoint(
    filepath=MODEL_CK_POINT_FPATH,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="auto",
)
callbacks_list = [checkpoint]

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)

model.fit(
    x=[training_img, train_padded_txt, train_input_length, train_label_length],
    y=np.zeros(len(training_img)),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(
        [valid_img, valid_padded_txt, valid_input_length, valid_label_length],
        [np.zeros(len(valid_img))],
    ),
    verbose=1,
    callbacks=callbacks_list,
)

print(
    model.evaluate(
        x=[training_img, train_padded_txt, train_input_length, train_label_length],
        y=np.zeros(len(training_img)),
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks_list,
    )
)


act_model.load_weights(MODEL_CK_POINT_FPATH)

# predict outputs on validation images
prediction = act_model.predict(valid_img[:10])

# use CTC decoder
out = K.get_value(
    K.ctc_decode(
        prediction,
        input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
        greedy=True,
    )[0][0]
)

# see the results
i = 0
for x in out:
    print("original_text =  ", valid_orig_txt[i])
    print("predicted text = ", end="")
    for p in x:
        if int(p) != -1:
            print(CHAR_LIST[int(p)], end="")
    print("\n")
    i += 1
