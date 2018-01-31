import numpy as np
from keras import Model
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.layers import Input, Conv2D, MaxPool2D, Flatten
from keras.utils.vis_utils import plot_model
import os

os.environ["PATH"] += os.pathsep + 'D:/DevelopTools/Graphviz2.38/bin/bin/'

batch_size = 128
nb_classes = 10
epoch = 20

(X_train, y_train), (X_test, y_test) = mnist.load_data('D:/GitCode/python/keras_L/mnist.npz')
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# model = Sequential()
# model.add(Dense(512, input_shape=(784,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Activation('softmax'))

inputs = Input(shape=(784,))
x = Dense(512)(inputs)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

x = Dense(512)(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

x = Dense(10)(x)
prediction = Activation('softmax')(x)

model = Model(inputs=inputs, outputs=prediction)

model.summary()
# plot_model(model, to_file='model.png')

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epoch,
                    verbose=1,
                    validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


model.save('data/mnist-mpl.h5')