from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import Input, Model, Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, Convolution2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np

nb_train_samples = 12
nb_validation_samples = 12
img_width = 150
img_height = 150
batch_size = 16
train_data_dir = 'D:/retrain/retrain_data_color/backpack/train_data/train_data'
validation_data_dir = 'D:/retrain/retrain_data_color/backpack/train_data/validation_data'

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(12))
model.add(Activation('softmax'))

model.summary()



model.load_weights('first_try.h5')

reshape = image.load_img(
    'D:/retrain/retrain_data_color/backpack/train_data/train_data/cyan/000000023.jpg',
    target_size=(150, 150))
img_data = image.img_to_array(reshape)
img_data = img_data * 1. / 255
x = np.expand_dims(img_data, axis=0)
b = model.predict(x)
print(b)
print(np.sum(b))

classes = np.argmax(b)
print(classes)

validation_data_dir = 'D:/retrain/retrain_data_color/backpack/train_data/validation_data'
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# generator = model.evaluate_generator(validation_generator, 5)
# print(generator)
