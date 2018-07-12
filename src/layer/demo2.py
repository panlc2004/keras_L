from keras.layers import Reshape, RepeatVector, Input
import numpy as np
import tensorflow as tf

x = [[1, 2, 3, 4, 5, 6], [11, 22, 33, 44, 55, 66]]
x = np.asarray(x)
x = tf.cast(x, tf.float32)
print(x.shape)
y = RepeatVector(2)(x)
print(y.shape)

sess = tf.Session()
y = sess.run(y)
print(y.shape)
print(y)
