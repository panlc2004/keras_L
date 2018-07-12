import tensorflow as tf

v1 = tf.Variable(tf.random_normal([6, 1], stddev=0.35), name="v1")
v2 = tf.Variable(tf.constant([3, 2, 1]), name="v2")
saver1 = tf.train.Saver([v1])
saver2 = tf.train.Saver([v2])
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver2.restore(sess, "checkpoint/model_test-1")
    saver1.restore(sess, "checkpoint/model_test-1")
    a = sess.run(v1)
    print(a)
    a = sess.run(v2)
    print(a)
