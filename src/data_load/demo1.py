import numpy as np
from  keras.layers import Input, K
from keras.models import Model
from keras.callbacks import Callback
import tensorflow as tf


class Print(Callback):
    def on_train_begin(self, logs={}):
        print('on_train_begin')
        print(logs)

    def on_batch_begin(self, batch, logs=None):
        print('on_batch_begin')
        print(logs)

    def on_epoch_begin(self, batch, logs={}):
        print('on_epoch_begin')
        print(logs)

    def on_epoch_end(self, batch, logs={}):
        print('on_epoch_end')
        print(logs)

    def on_batch_end(self, batch, logs={}):
        print('on_batch_end')
        print(logs)

    def on_train_end(self, batch, logs={}):
        print('on_train_end')
        print(logs)


print(K.image_data_format())

x = Input(shape=(1,))
y = tf.constant(6.0, tf.float32, name='accuracy')
model = Model(inputs=x, outputs=x)
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

data = np.random.random((3, 1))
labels = np.random.randint(2, size=(3, 1))
labels.reshape([3, -1])

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(1).repeat(100000)
iterator = dataset.make_one_shot_iterator()
x, y = iterator.get_next()


def generate_arrays_from_file():
    while True:
        yield data, labels


# Train the model, iterating on the data in batches of 32 samples
model.fit_generator(generate_arrays_from_file(), steps_per_epoch=1, epochs=50, callbacks=[Print()])
