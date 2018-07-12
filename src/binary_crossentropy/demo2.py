from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
import numpy as np
import tensorflow as tf

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

inputs = Input(shape=(7,), name='input')
x = Dense(12, activation='relu', name='x1')(inputs)
x = Dense(8, activation='relu', name='x2')(x)
x = Dense(1, activation='sigmoid', name='x3')(x)

modal = Model(inputs, x)


x_data = np.random.rand(5000, 7)
y_data = np.random.randint(0, 2, [5000, 1])

modal.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

modal.fit(x_data, y_data, epochs=500, batch_size=200)


# def gen():
#     while True:
#         x_data = np.random.rand(5000, 7)
#         y_data = np.random.randint(0, 2, [5000, 1])
#         yield x_data, y_data
#
# modal.fit_generator(gen, steps_per_epoch=50)




# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     r = sess.run(x, feed_dict={inputs: a})
#     print(r)
#     modal.save_weights('./model_weight/weight.h5')
#     modal.save('./model/model1.h5')
