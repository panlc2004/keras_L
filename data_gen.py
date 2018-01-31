from keras.preprocessing.image import ImageDataGenerator
from keras import Input, Model, Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, Convolution2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

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

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

train_nums = train_generator.samples
print(train_nums)
print(train_generator.class_indices)
print('++++++++++++++++++++++')


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_nums = validation_generator.samples
print(validation_generator.class_indices)


# 如果说训练样本树N=1000，steps_per_epoch = 10，那么相当于一个batch_size=100
model.fit_generator(
    train_generator,
    steps_per_epoch=train_nums // batch_size // 10,
    nb_epoch=1,
    validation_data=validation_generator,
    validation_steps=test_nums // batch_size)

# model.save_weights('first_try2.h5')