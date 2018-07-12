import numpy as np
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Lambda, Input
from keras.models import Model
from keras.optimizers import Adam
from keras_preprocessing import image
from keras.callbacks import TensorBoard, ModelCheckpoint


batch_size = 50
# 学习速率调整速度
img_size = [96, 96]


def similarity(x):
    y1, y2 = tf.unstack(x, 2, 1)
    # 计算两张图片相似度 : x^2 相似度
    n = tf.square(tf.subtract(y1, y2))
    m = tf.add(y1, y2)
    x = tf.divide(n, m)
    return x


def build_model(input_img_size, embedding_size):
    inputs = Input((2, input_img_size[0], input_img_size[1], 3), name='group_input')
    inputs_reshape = Lambda(lambda x: tf.reshape(x, [batch_size * 2, input_img_size[0], input_img_size[1], 3]),
                            name='group_input_reshape')(inputs)
    base_module = InceptionResNetV2(weights=None, input_tensor=inputs_reshape, classes=embedding_size)
    custom_input = base_module.output
    # 图片总数量 = batch_size * 2
    x = Lambda(lambda x: tf.reshape(x, [batch_size, 2, embedding_size]), name='prediction_reshape')(custom_input)
    # 矩阵相似计算：x^2相似
    # x = Lambda(similarity, input_shape=[2, embedding_size], name='similarity')(x)
    # # 线性回归
    # x = Dense(1, activation='sigmoid', name='final_predict')(x)
    model = Model(base_module.input, x)
    return model


model = build_model(img_size, 128)

model.load_weights('face_model_epoch_51.h5',by_name=True)

# tensorboard = TensorBoard(log_dir='D:/panlc/Code/factnet_train/log', batch_size=batch_size,
#                           histogram_freq=0, write_graph=True, write_images=False)
# checkpoint = ModelCheckpoint("D:/panlc/Code/factnet_train/model/face_model_epoch_{epoch}.h5", monitor='val_loss',
#                              save_weights_only=True, save_best_only=True)
#
# model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00003), metrics=['accuracy'])
# model.fit_generator(train_gen,
#                     steps_per_epoch=train_loader.batch_num,
#                     epochs=2000,
#                     callbacks=[tensorboard, checkpoint],
#                     validation_data=validate_data,
#                     initial_epoch=48)