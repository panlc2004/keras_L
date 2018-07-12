from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

inputs = Input(shape=(8,), name='input')
# x1 = Dense(12, activation='relu', name='x1')(inputs)
# x2 = Dense(8, activation='relu', name='x2')(x1)
x3 = Dense(1, activation='sigmoid', name='x3')(inputs)

# variables_to_restore = slim.get_variables_to_restore(include=["x3"])
saver = tf.train.Saver()

a = np.asarray([[0.13494128, 0.44912577, 0., 0.51827312, 0., 0.31411514, 0.55605394, 0.],
                [0.22719835, 0., 0.3285324, 0., 0., 0., 0., 0.54578739]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'ckpt/test')
    r = sess.run(x3, feed_dict={inputs: a})
    print(r)
