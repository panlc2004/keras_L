from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model, load_model
import numpy
import tensorflow as tf

# fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)


modal = load_model('./model/model1.h5')

import numpy as np

a = np.zeros([2, 7])
a[0] = a[0] - 1
a[1] = a[1] + 1
print(a)

modal.predict(a)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     r = sess.run(modal.output, feed_dict={modal.input: a})
#     print(r)
