from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


inputs = Input(shape=(7,), name='input')
x1 = Dense(12, activation='relu', name='x1')(inputs)
x2 = Dense(8, activation='relu', name='x2')(x1)
# x3 = Dense(1, activation='sigmoid', name='x3')(x2)

variables_to_restore = slim.get_variables_to_restore(include=["x1", "x2"])
saver = tf.train.Saver(variables_to_restore)

a = np.zeros([2, 7])
a[0] = a[0] - 1
a[1] = a[1] + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'ckpt/test')
    r = sess.run(x2, feed_dict={inputs: a})
    print(r)
