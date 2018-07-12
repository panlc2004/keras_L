# x3/Sigmoid:0
import numpy as np
from keras.models import load_model
import tensorflow as tf
# from keras import backend as K



a = np.zeros([2, 7])
a[0] = a[0] - 1
a[1] = a[1] + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model = load_model('model.h5', compile=False)

    print(1)
    #不加有时会报错，有时不会
    # K.set_session(sess)
    input = sess.graph.get_tensor_by_name('input:0')
    out = sess.graph.get_tensor_by_name('x3/Sigmoid:0')
    r = sess.run(out,feed_dict={input:a})
    print(r)

