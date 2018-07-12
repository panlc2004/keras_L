import tensorflow as tf

inp = tf.placeholder(tf.int32, shape=[3, 1])
v2 = tf.Variable(tf.constant([1, 2, 2]), name="v2")
pre = tf.add(inp, v2)

# saver2 = tf.train.Saver([v2])
saver2 = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver2.restore(sess, "checkpoint/model_test-1")
    v2 = sess.run(v2)
    print(v2)

    import numpy as np

    ii = np.asarray([1, 2, 1])
    print(ii.shape)
    ii = np.reshape(ii, [3, 1])
    p = sess.run(pre, feed_dict={inp: ii})
    print(p)
