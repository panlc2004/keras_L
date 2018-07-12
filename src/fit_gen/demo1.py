import csv

import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten
from keras.models import Model
from keras import optimizers
from keras_preprocessing import image


#
def generate_batch_data_random():
    """逐步提取batch数据到显存，降低对显存的占用"""
    while (True):
        yield np.asarray([[1.], [2.], [3.], [4.]]), np.asarray([[1.1], [2.1], [3.1], [4.1]])


def gen_data(target_size=(96, 96)):
    list = get_file_from_csv()
    for i in list:
        get_img_data(i, target_size=(96, 96))


def get_file_from_csv():
    filename = 'train_data.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        list = list(reader)
        np.random.shuffle(list)
        return list


def get_img_data(csv_data, target_size):
    img_path_group = get_img_path(csv_data)
    img1 = load_img(img_path_group[0], target_size)
    img2 = load_img(img_path_group[1], target_size)
    return img1, img2


def load_img(img_path, target_size=None):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img) / 255.
    return img


def get_img_path(csv_data):
    s = csv_data[0].replace("b'", "").replace("[", "").replace("]", "")
    s = s.split(" ")
    return s


# inputs = tf.placeholder(tf.float32, [None, 1])
inputs = Input(shape=(1,))
x = inputs
model = Model(inputs, x)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit_generator(generate_batch_data_random(),
                    samples_per_epoch=2,
                    nb_epoch=2,
                    )
