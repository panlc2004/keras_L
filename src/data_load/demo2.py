import tensorflow as tf
import numpy as np


def cal_cos(enm1, enm2):
    """
    计算两向量余弦相似度
    :param enm1:
    :param enm2:
    :return:
    """
    with tf.variable_scope('cal_cos'):
        mul = tf.multiply(enm1, enm2, name='mul')
        enm1_square = tf.square(enm1, name='enm1_square')
        enm2_square = tf.square(enm2, name='enm2_square')
        enm1_square_sum = tf.reduce_sum(enm1_square, 1, name='enm1_square_sum')
        enm2_square_sum = tf.reduce_sum(enm2_square, 1, name='enm2_square_sum')
        sqrt = tf.sqrt(enm1_square_sum * enm2_square_sum, name='sqrt')
        mul_reduce_sum = tf.reduce_sum(mul, 1, name='mul_reduce_sum')
        cos = tf.divide(mul_reduce_sum, sqrt, name='cos')
        return cos


def cos2(x3, x4):
    # 求模
    x3_norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=1))
    x4_norm = tf.sqrt(tf.reduce_sum(tf.square(x4), axis=1))
    # 内积
    x3_x4 = tf.reduce_sum(tf.multiply(x3, x4), axis=1)
    cosin = x3_x4 / (x3_norm * x4_norm)
    cosin1 = tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))
    return cosin1


anchor = np.zeros([9, 1000])

anchor[0] = anchor[0] + 0.1
anchor[1] = anchor[1] + 1
anchor[2] = anchor[2] + 2
anchor[3] = anchor[3] + 3
anchor[4] = anchor[4] + 4
anchor[5] = anchor[5] + 5
anchor[6] = anchor[6] + 6
anchor[7] = anchor[7] + 7
anchor[8] = anchor[8] + 8
print(anchor)

embeddings = tf.cast(anchor, tf.float32)
re = tf.reshape(embeddings, [-1, 3, 1000])
ree = re[0]
anchor1, positive, negative = tf.unstack(re, 3, 1)
cp = cal_cos(anchor1, positive)
cn = cal_cos(anchor1, negative)
basic_loss = tf.subtract(1.2, cp)
loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

cp1 = cos2(anchor1, positive)
cn1 = cos2(anchor1, negative)
aa = tf.cos(anchor1)



with tf.Session() as sess:
    print('======================')
    an = sess.run(anchor1)
    print(an)
    print('======================')
    bn = sess.run(positive)
    print(bn)
    print('======================')
    print(sess.run(negative))
    print('cos:', sess.run(aa))
    print(sess.run(cp))
    # print(sess.run(cn))
    print('======================')
    # print(sess.run(cp1))
    # print(sess.run(cn1))
    # print(sess.run(basic_loss))
    # print(sess.run(loss))
    # print(sess.run(re))
    # print(sess.run(ree))
    # num=float(np.sum(an*bn))
    # denom=np.linalg.norm(an)*np.linalg.norm(bn)
    # cos=num/denom
    # print(cos)