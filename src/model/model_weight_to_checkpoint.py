from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import numpy as np

inputs = Input(shape=(7,), name='input')
x1 = Dense(12, activation='relu', name='x1')(inputs)
x2 = Dense(8, activation='relu', name='x2')(x1)
x3 = Dense(1, activation='sigmoid', name='x3')(x2)
modal = Model(inputs, x3)

a = np.zeros([2, 7])
a[0] = a[0] - 1
a[1] = a[1] + 1

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    modal.load_weights('./model_weight.h5')
    r = sess.run(x3, feed_dict={inputs: a})
    print(r)
    saver.save(sess, 'ckpt/test')
