from keras.datasets import mnist
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import np_utils

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

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = load_model('data/mnist-mpl.h5')

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

