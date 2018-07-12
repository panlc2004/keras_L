import tensorflow as tf

v1= tf.Variable(tf.random_normal([6, 1], stddev=0.35), name="v1")
v2= tf.Variable(tf.zeros([3]), name="v2")
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver.save(sess,"checkpoint/model_test",global_step=1)