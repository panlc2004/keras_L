import numpy as np
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Lambda, Input
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing import image
from keras.callbacks import TensorBoard, ModelCheckpoint

batch_size = 1
# 学习速率调整速度
img_size = [96, 96]


def similarity(x):
    y1, y2 = tf.unstack(x, 2, 1)
    # 计算两张图片相似度 : x^2 相似度
    n = tf.square(tf.subtract(y1, y2))
    m = tf.add(y1, y2)
    x = tf.divide(n, m)
    return x


embedding_size = 128

# inputs = tf.placeholder(tf.float32, [None, 96, 96, 3], name='input')
inputs = Input((img_size[0], img_size[1], 3), name='group_input')

base_module = InceptionResNetV2(weights=None, input_tensor=inputs, classes=embedding_size)
custom_input = base_module.output
# 图片总数量 = batch_size * 2
x = Lambda(lambda x: tf.reshape(x, [batch_size, 2, embedding_size]), name='prediction_reshape')(custom_input)
# 矩阵相似计算：x^2相似
x = Lambda(similarity, input_shape=[2, embedding_size], name='similarity')(x)
# # 线性回归
x = Dense(1, activation='sigmoid', name='final_predict')(x)
model = Model(base_module.input, x)

model.load_weights('face_model_epoch_19.h5')


def get_img_data(img_path):
    img = image.load_img(img_path, target_size=[96, 96])
    img = image.img_to_array(img) / 255.
    return img


img1 = get_img_data('F:/facenet_train_data/train/person0/person0_0.png')
img2 = get_img_data('F:/facenet_train_data/train/person0/person0_1.png')
img = [img1, img2]
# img = [img]
img = np.asarray(img)
print(img.shape)
predict = model.predict(img)
print(predict)
