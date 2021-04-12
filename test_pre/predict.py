import numpy as np
import keras
import keras.backend as K

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


from utils import CHAR_LIST, encode_to_labels, read_img


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
act_model.load_weights("best_model.hdf5")

im_test = read_img(r"C:\Users\quock\Desktop\test1.jpg")
valid_img = np.array([im_test])

# predict outputs on validation images
prediction = act_model.predict(valid_img)


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
    #print("original_text =  ", valid_orig_txt[i])
    print("predicted text = ", end = '')
    for p in x:  
        if int(p) != -1:
            print(CHAR_LIST[int(p)], end = '')       
    print('\n')
    i+=1
