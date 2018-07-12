from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras import backend as K

import tensorflow as tf
import numpy as np

base_module = InceptionResNetV2(weights=None)

custom_input = base_module.output
x = Dropout(0.8, name='dropout_1')(custom_input)
x = Dense(768, activation='relu')(x)
x = Dropout(0.8, name='dropout_2')(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.8, name='dropout_3')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.8, name='dropout_4')(x)
x = Dense(1, activation='sigmoid', name='prediction')(x)

# custom_model = Model(base_module.input, x)
# optimizer = SGD(lr=0.2, momentum=0.9)
# custom_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# custom_model.fit(x_data, y_data, batch_size=2, epochs=1)

# y = tf.placeholder(tf.float32, [None, 2], name='y')
# lo = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=x)
# loss = tf.reduce_mean(lo)

y = tf.placeholder(tf.float32, [None, 1], name='y')
lo2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=x)
loss2 = tf.reduce_mean(lo2)

init = tf.global_variables_initializer()

x_data = np.random.rand(2, 299, 299, 3)
y_data = np.asarray([[1, 0], [0, 1]])
y_data_2 = np.asarray([[1], [0]])

with tf.Session() as sess:
    # 重要
    K.set_session(sess)
    sess.run(init)
    y_pre = sess.run(x, feed_dict={base_module.input: x_data})
    print(y_pre)
    print("==========")
    lo = sess.run(lo2, feed_dict={base_module.input: x_data, y: y_data_2})
    print(lo)
    l = sess.run(loss2, feed_dict={base_module.input: x_data, y: y_data_2})
    print(l)
