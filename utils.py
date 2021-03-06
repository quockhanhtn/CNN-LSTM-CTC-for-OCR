import string
import cv2
import numpy as np

# char_list:   'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
CHAR_LIST = string.ascii_letters + string.digits


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(CHAR_LIST.index(char))
        except Exception:
            print(char)

    return dig_lst


def read_img(img_file_path):
    # read input image and convert into gray scale image
    img = cv2.cvtColor(cv2.imread(img_file_path), cv2.COLOR_BGR2GRAY)

    # convert each image of shape (32, 128, 1)
    w, h = img.shape
    if h > 128 or w > 32:
        return None
    if w < 32:
        add_zeros = np.ones((32 - w, h)) * 255
        img = np.concatenate((img, add_zeros))

    if h < 128:
        add_zeros = np.ones((32, 128 - h)) * 255
        img = np.concatenate((img, add_zeros), axis=1)
    img = np.expand_dims(img, axis=2)

    # Normalize each image
    img = img / 255.0
    return img


def resize_img(img_arr):
    w, h = img_arr.shape

    if h > 128 or w > 32:
        img_arr = cv2.resize(img_arr, dsize=(128, 32), interpolation=cv2.INTER_CUBIC)
        return np.expand_dims(img_arr, axis=2)

    if w < 32:
        add_zeros = np.ones((32 - w, h)) * 255
        img_arr = np.concatenate((img_arr, add_zeros))

    if h < 128:
        add_zeros = np.ones((32, 128 - h)) * 255
        img_arr = np.concatenate((img_arr, add_zeros), axis=1)

    img_arr = np.expand_dims(img_arr, axis=2)

    # Normalize each image
    img_arr = img_arr / 255.0
    return img_arr