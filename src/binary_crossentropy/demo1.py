from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
import numpy
import tensorflow as tf

# fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)

inputs = Input(shape=(7,), name='input')
x = Dense(12, activation='relu', name='x1')(inputs)
x = Dense(8, activation='relu', name='x2')(x)
x = Dense(1, activation='sigmoid', name='x3')(x)

modal = Model(inputs, x)
modal.summary()

import numpy as np

a = np.zeros([2, 7])
a[0] = a[0] - 1
a[1] = a[1] + 1
print(a)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    r = sess.run(x, feed_dict={inputs:a})
    print(r)
    modal.save_weights('./model_weight/weight.h5')
    modal.save('./model/model1.h5')
